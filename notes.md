# Main PPO
```
trainer.main_ppo (工厂，初始化trainer并绑定worker到trainer)
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
   }
```
```
trainer.ppo.ray_trainer.init_workers() (组装worker协作，训练抽象流程）
(reward_model=False, rule-based reward)
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
{
    'prompts': prompt,
    'responses': response,
    'input_ids': seq,
    'attention_mask': attention_mask,
    'position_ids': position_ids
}

ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
{
    'ref_log_prob': dp_actor.compute_log_prob()
}

values = self.critic_wg.compute_values(batch)

reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor

batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)

batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam, num_repeat=self.config.actor_rollout_ref.rollout.n)

critic_output = self.critic_wg.update_critic(batch)
actor_output = self.actor_rollout_wg.update_actor(batch)
self._validate()
```
```
--workers.fsdp_workers._build_model_optimizer() (初始化所有worker)
1. ActorRolloutRefWorker
----workers.dp_actor (share actor_module)
    use_remove_padding = False
    output = self.actor_module(input_ids=input_ids_rmpad,
                                attention_mask=None/attention,
                                position_ids=position_ids_rmpad,
                                use_cache=False)  # prevent model thinks we are generating
logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
----ref (load a copy model), same behavior like actor
----workers.hf_rollout (share actor_module)
    with param_ctx:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                do_sample=do_sample,
                max_new_tokens=response_length,
                # max_length=max_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                # renormalize_logits=True,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True)
    seq = output.sequences

2. CriticWorker
----workers.dp_critic
    output = self.critic_module(input_ids=input_ids_rmpad,
                                attention_mask=None,
                                position_ids=position_ids_rmpad,
                                use_cache=False)  # prevent model thinks we are generating
```

# Ratio KL Inference
```
ray_trainier.py (train_batch_size)
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    batch = batch.union(gen_batch_output)
    actor_output = self.actor_rollout_wg.update_actor(batch)

fsdp_workers.py
    def generate_sequences(self, prompts: DataProto):
        output = self.rollout.generate_sequences(prompts=prompts)
        old_log_probs = self.actor.compute_log_prob(data=output)
        output.batch['old_log_probs'] = old_log_probs

    def update_actor(self, data: DataProto):
        metrics = self.actor.update_policy(data=data)

hf_rollout.py (micro_batch_size)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        output = [self._generate_minibatch(p) for p in batch_prompts]

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        prob_token = self.module.generate()
        seq = prob_token.sequences
        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'prob_tokens': prob_token,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

dp_actor.py (micro_batch_size)
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)

    def update_policy(self, data: DataProto):
        (ppo_mini_batch_size for loss.backward())
        dataloader = batch.split(self.config.ppo_mini_batch_size)
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages)
            loss = policy_loss / self.gradient_accumulation
            loss.backward()
            self.actor_optimizer.step()

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.actor_module(input_ids=prob_tokens)
        logits = output.logits
        log_probs = logprobs_from_logits(logits, micro_batch['responses'])
```
# Batch Size
```
training_bs 256
ray_trainer.py
    reward - refKL penalty

mini_bs 128
dp_critic.py
_optimizer_step()
dp_actor.py
_optimizer_step()

micro_bs 4
hf_rollout.py
    _generate_minibatch()
dp_critic.py
    _forward_micro_batch()
    loss.backward()
dp_actor.py
    ratio_KL
    _forward_micro_batch()
    loss.backward()
```

# Reward Function
```
verl/trainer/main_ppo.py 160
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

verl/trainer/ppo/reward.py 60
    reward_manager_name = config.reward_model.get("reward_manager", "naive")

verl/workers/reward_manager/naive.py 40
    self.compute_score = compute_score or default_compute_score

verl/utils/reward_score/__init__.py 
    if data_source == "openai/gsm8k": pass
```

# Data Loading
```
verl/trainer/main_ppo.py 167
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)

    verl/utils/dataset/rl_dataset.py 68 RLHFDataset()
        def _read_files_and_tokenize(self): filter length

-> verl/trainer/ppo/ray_trainer.py 
class RayPPOTrainer: 483
    self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler) -> self.train_dataloader, self.val_dataloader

    def fit(self): 905
        val_metrics = self._validate()

    def _validate(self): 599
        for test_data in self.val_dataloader:
            input_ids = test_batch.batch["input_ids"]

    verl/utils/dataset/rl_dataset.py 212
        def __getitem__(self, item):
```