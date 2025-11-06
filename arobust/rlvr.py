import asyncio
import json
import os
import uuid
from typing import Any, Dict

import torch
from tensordict import NonTensorData, TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.padding import concat_padded_tensors
from areal.utils.util import worker_dump_rollout_output
from realhf.api.core.data_api import RL_TASKS, load_hf_tokenizer
from realhf.base import logging

logger = logging.getLogger("RLVRWorkflow")


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast = None,
        tokenizer_path: str = None,
        exp_name: str = None,
        trial_name: str = None,
    ):
        if tokenizer is None and tokenizer_path is None:
            raise ValueError("Either tokenizer or tokenizer_path must be provided")

        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer if tokenizer is not None else None
        self.tokenizer_path = tokenizer_path
        self._step = None
        self._rank = None
        self.exp_name = exp_name
        self.trial_name = trial_name

    async def _compute_reward(
        self, prompt, completion, prompt_ids, completion_ids, stop_reason, data
    ):
        """计算 reward"""
        return await self.reward_fn(
            prompt, completion, prompt_ids, completion_ids, **data
        )

    async def _run_new_prompt_task(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        Run a new prompt task (no previous_ids).
        """
        data_name_id = f"q[{data['query_id'][0]}]i[{data['index_in_group'][0]}]"

        logger.info(
            f"[RLVRWorkflow] data {data_name_id} run_new_prompt_task with data: {data}"
        )
        assert data.get("prompt") is not None
        assert data.get("previous_ids") is None

        prompt_text = data["prompt"][0]
        input_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )["input_ids"]

        new_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.gconfig.max_new_tokens,
            max_tokens=self.gconfig.max_tokens,
            min_new_tokens=self.gconfig.min_new_tokens,
            greedy=self.gconfig.greedy,
            top_p=self.gconfig.top_p,
            top_k=self.gconfig.top_k,
            temperature=self.gconfig.temperature,
            stop_token_ids=self.gconfig.stop_token_ids,
        )
        rid = uuid.uuid4().hex
        req = LLMRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=new_gconfig,
        )
        logger.info(f"[RLVRWorkflow] data {data_name_id} start to agenerate")
        resp = await engine.agenerate(req)
        logger.info(f"[RLVRWorkflow] data {data_name_id} agenerate done")

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0] * resp.input_len + resp.output_logprobs
        prompt_mask = [1] * resp.input_len + [0] * resp.output_len
        output_version = resp.output_version
        versions = [output_version] * (resp.input_len + resp.output_len)

        # seq_no_eos_mask，有 EOS 是 False，没有 EOS 是 True
        seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
            seq[-1] != self.tokenizer.pad_token_id
        )

        if "prompt" in data.keys():
            del data["prompt"]

        completion = self.tokenizer.decode(
            resp.output_tokens,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )

        logger.info(f"[RLVRWorkflow] data {data_name_id} start to compute reward")
        reward = await self._compute_reward(
            prompt=prompt_text,
            completion=completion,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            stop_reason=resp.stop_reason,
            data=data,
        )
        logger.info(f"[RLVRWorkflow] data {data_name_id} compute reward done")

        task_id = RL_TASKS.index(data["task"][0])
        sample_info = {
            "prompt": prompt_text,
            "completion": completion,
            "reward": reward,
            "solutions": data.get("solutions", []),
            "task": data["task"][0],
            "task_id": task_id,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "output_logprobs": resp.output_logprobs,
            "seq_len": len(seq),
            "versions": versions,
            "query_id": data["query_id"][0],
            "stop_reason": resp.stop_reason,
        }

        worker_dump_rollout_output(sample_info=sample_info)

        res = dict(
            input_ids=torch.tensor(seq).unsqueeze(0),
            prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq)).unsqueeze(0),
            rewards=torch.tensor([reward]),
            seqlen=torch.tensor([len(seq)]),
            task_ids=torch.tensor([task_id]),
            seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            query_id=NonTensorData([data["query_id"][0]]),
            index_in_group=NonTensorData([data["index_in_group"][0]]),
            original_data=NonTensorData(data),
        )
        return TensorDict(res, batch_size=[1])

    async def _run_reapply_task(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        Run a reapply task (with previous_ids, for continuation).
        """
        data_name_id = f"q[{data['query_id'][0]}]i[{data['index_in_group'][0]}]"
        logger.info(
            f"[RLVRWorkflow] data {data_name_id} run_reapply_task with data: {data}"
        )
        assert data.get("previous_ids") is not None

        input_ids = data["previous_ids"][0]
        prompt_len = data["previous_prompt_len"][0]
        prompt_ids = input_ids[:prompt_len]
        prompt_text = self.tokenizer.decode(
            prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )

        # seq_no_eos_mask，有 EOS 是 False，没有 EOS 是 True
        is_sample_finished = (not data["previous_seq_no_eos_mask"][0]) or (
            len(input_ids) >= self.gconfig.max_tokens
        )

        if not is_sample_finished:
            # 续推
            new_gconfig = self.gconfig.new(
                n_samples=1,
                max_new_tokens=self.gconfig.max_new_tokens,
                max_tokens=self.gconfig.max_tokens,
                min_new_tokens=self.gconfig.min_new_tokens,
                greedy=self.gconfig.greedy,
                top_p=self.gconfig.top_p,
                top_k=self.gconfig.top_k,
                temperature=self.gconfig.temperature,
                stop_token_ids=self.gconfig.stop_token_ids,
            )
            rid = uuid.uuid4().hex
            req = LLMRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=new_gconfig,
            )
            logger.info(f"[RLVRWorkflow] data {data_name_id} start to agenerate (reapply)")
            resp = await engine.agenerate(req)
            logger.info(f"[RLVRWorkflow] data {data_name_id} agenerate done (reapply)")

            seq = resp.input_tokens + resp.output_tokens

            completion_len = len(seq) - prompt_len
            completion_ids = seq[prompt_len:]

            previous_logprobs = data["previous_logprobs"][0][
                : resp.input_len
            ]  # 使用 input_len 而不是 prompt_len，因为可能是中间打断的
            logprobs = previous_logprobs + resp.output_logprobs
            prompt_mask = [1] * prompt_len + [0] * completion_len
            output_versions = [resp.output_version] * resp.output_len
            versions = (
                data["previous_version"][0][: resp.input_len] + output_versions
            )
            seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
                seq[-1] != self.tokenizer.pad_token_id
            )

            stop_reason = resp.stop_reason
        else:
            # sample is already finished, we do not need to request a new rollout
            seq = input_ids
            completion_len = len(seq) - prompt_len
            completion_ids = seq[prompt_len:]

            logprobs = data["previous_logprobs"][0]
            prompt_mask = [1] * prompt_len + [0] * completion_len
            versions = data["previous_version"][0]
            seq_no_eos_mask = data["previous_seq_no_eos_mask"][0]

            stop_reason = "length" if seq_no_eos_mask else "stop"

        if "prompt" in data.keys():
            del data["prompt"]
        if "previous_ids" in data.keys():
            del data["previous_ids"]
        if "previous_version" in data.keys():
            del data["previous_version"]
        if "previous_logprobs" in data.keys():
            del data["previous_logprobs"]

        completion = self.tokenizer.decode(
            completion_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )

        logger.info(f"[RLVRWorkflow] data {data_name_id} start to compute reward")
        if not is_sample_finished:
            reward = await self._compute_reward(
                prompt=prompt_text,
                completion=completion,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                stop_reason=stop_reason,
                data=data,
            )
        else:
            reward = data["previous_rewards"][0]
        logger.info(f"[RLVRWorkflow] data {data_name_id} compute reward done")

        task_id = RL_TASKS.index(data["task"][0])

        sample_info = {
            "prompt": prompt_text,
            "completion": completion,
            "reward": reward,
            "solutions": data.get("solutions", []),
            "task": data["task"][0],
            "task_id": task_id,
            "input_tokens": seq[:prompt_len],
            "output_tokens": seq[prompt_len:],
            "output_logprobs": logprobs[prompt_len:],
            "seq_len": len(seq),
            "versions": versions,
            "query_id": data["query_id"][0],
            "stop_reason": stop_reason,
        }

        worker_dump_rollout_output(sample_info=sample_info)

        res = dict(
            input_ids=torch.tensor(seq).unsqueeze(0),
            prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq)).unsqueeze(0),
            rewards=torch.tensor([reward]),
            seqlen=torch.tensor([len(seq)]),
            task_ids=torch.tensor([task_id]),
            seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            query_id=NonTensorData([data["query_id"][0]]),
            index_in_group=NonTensorData([data["index_in_group"][0]]),
            original_data=NonTensorData(data),
        )
        return TensorDict(res, batch_size=[1])

    async def arun_episode(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        主入口：根据是否有 previous_ids 判断是新 prompt 还是续推任务
        """
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)

        if data.get("previous_ids") is None:
            # 新 prompt
            return await self._run_new_prompt_task(engine, data)
        else:
            # 续推
            return await self._run_reapply_task(engine, data)

    async def arun_episode_old(self, engine, data):
        """保留原始实现作为参考"""
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        text = data["prompt"][0]
        prompt_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        n_samples = self.gconfig.n_samples
        new_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.gconfig.max_new_tokens,
            max_tokens=self.gconfig.max_tokens,
            min_new_tokens=self.gconfig.min_new_tokens,
            greedy=self.gconfig.greedy,
            top_p=self.gconfig.top_p,
            top_k=self.gconfig.top_k,
            temperature=self.gconfig.temperature,
            stop_token_ids=self.gconfig.stop_token_ids,
        )

        req = LLMRequest(
            rid=uuid.uuid4().hex,
            text=text,
            input_ids=prompt_encodings["input_ids"],
            gconfig=new_gconfig,
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
        results = []

        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            output_version = resp.output_version
            versions = [output_version] * (resp.input_len + resp.output_len)
            seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
                seq[-1] != self.tokenizer.pad_token_id
            )

            if "prompt" in data.keys():
                del data["prompt"]

            completion = self.tokenizer.decode(
                resp.output_tokens,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )

            reward = await self.reward_fn(
                prompt=text,
                completion=completion,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            task_id = RL_TASKS.index(data["task"][0])

            sample_info = {
                "prompt": text,
                "completion": completion,
                "reward": reward,
                "solutions": data.get("solutions", []),
                "task": data["task"][0],
                "task_id": task_id,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "output_logprobs": resp.output_logprobs,
                "seq_len": len(seq),
                "versions": versions,
                "query_id": data["query_id"][0],
                "stop_reason": resp.stop_reason,
            }

            worker_dump_rollout_output(sample_info=sample_info)

            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(
                    0
                ),  # seq=[10, 22, 33] => tensor([[10, 22, 33]])
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                rewards=torch.tensor([reward]),
                seqlen=torch.tensor([len(seq)]),
                task_ids=torch.tensor([task_id]),
                seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)

    async def arun_episodes(self, engine, data_list):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)

        n_samples = self.gconfig.n_samples
        new_gconfig = self.gconfig.new(
            n_samples=n_samples,
            max_new_tokens=self.gconfig.max_new_tokens,
            max_tokens=self.gconfig.max_tokens,
            min_new_tokens=self.gconfig.min_new_tokens,
            greedy=self.gconfig.greedy,
            top_p=self.gconfig.top_p,
            top_k=self.gconfig.top_k,
            temperature=self.gconfig.temperature,
            stop_token_ids=self.gconfig.stop_token_ids,
        )

        reqs = []
        texts = []
        for data in data_list:
            text = data["prompt"][0]
            texts.append(text)
            prompt_encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=None,
                padding=False,
                return_length=True,
                return_attention_mask=False,
            )

            req = LLMRequest(
                rid=uuid.uuid4().hex,
                text=text,
                input_ids=prompt_encodings["input_ids"],
                gconfig=new_gconfig,
            )
            reqs.append(req)

        resps = await engine.agenerate_batch(reqs)
        results = []

        for index, resp in enumerate(resps):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            output_version = resp.output_version
            versions = [output_version] * (resp.input_len + resp.output_len)
            seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
                seq[-1] != self.tokenizer.pad_token_id
            )
            data = data_list[index // new_gconfig.n_samples]

            completion = self.tokenizer.decode(
                resp.output_tokens,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )

            text = texts[index // new_gconfig.n_samples]
            if "prompt" in data.keys():
                del data["prompt"]

            reward = await self.reward_fn(
                prompt=text,
                completion=completion,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )

            task_id = RL_TASKS.index(data["task"][0])

            sample_info = {
                "prompt": text,
                "completion": completion,
                "reward": reward,
                "solutions": data.get("solutions", []),
                "task": data["task"][0],
                "task_id": task_id,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "output_logprobs": resp.output_logprobs,
                "seq_len": len(seq),
                "versions": versions,
                "query_id": data["query_id"][0],
                "stop_reason": resp.stop_reason,
            }

            worker_dump_rollout_output(sample_info=sample_info)

            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(
                    0
                ),  # seq=[10, 22, 33] => tensor([[10, 22, 33]])
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                rewards=torch.tensor([reward]),
                seqlen=torch.tensor([len(seq)]),
                task_ids=torch.tensor([task_id]),
                seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            )
            results.append(TensorDict(res, batch_size=[1]))
        return concat_padded_tensors(results)
