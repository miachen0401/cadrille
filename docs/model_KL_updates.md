# ==========================================
# ARLArena (GRPO/SAMPO) 训练与诊断伪代码
# ==========================================

for step in range(total_steps):
    
    # ---------------------------------------------------------
    # 阶段 1：Rollout 阶段 (采样、评估、固定 Advantage)
    # ---------------------------------------------------------
    with torch.no_grad():
        # 1. 使用当前策略 (记作 old_policy) 生成一批完整的 CAD 代码或推理轨迹
        trajectories, old_logits = old_policy.generate(prompts)
        old_log_probs = get_log_probs(old_logits, trajectories)
        
        # 2. 将生成的轨迹送入环境验证器，获取客观分数 (例如几何 IoU、能否编译)
        rewards = environment.get_rewards(trajectories)
        
        # 3. 计算优势函数 (Advantage)。
        # 在 GRPO 中，通常是同一 Prompt 下的奖励减去该组的均值，再除以标准差。
        # 注意：一旦这里算完，Advantage 就彻底固定了！它代表环境对轨迹的盖棺定论。
        advantages = compute_group_relative_advantage(rewards)


    # ---------------------------------------------------------
    # 阶段 2：Optimization 阶段 (计算 IS，算 Loss，更新参数)
    # ---------------------------------------------------------
    # 通常会对同一批采样数据循环优化几个 Epoch
    for epoch in range(ppo_epochs):
        
        # 1. 用正在更新的模型 (current_policy) 再次前向传播同一批轨迹
        new_logits = current_policy.forward(trajectories)
        new_log_probs = get_log_probs(new_logits, trajectories)
        
        # 2. 计算重要性采样比率 (IS Ratio)
        # 数学上 IS = pi_new / pi_old，对数空间下为 exp(new - old)
        is_ratios = torch.exp(new_log_probs - old_log_probs)
        
        # 3. 计算 Policy Loss
        # 这一步必须把动态的 IS 和固定的 Advantage 乘起来。
        # 传统的做法是做 Token 级别的 Clipping，而 SAMPO 在这里引入了序列级裁剪和掩码
        surrogate_loss = is_ratios * advantages
        clipped_loss = torch.clip(is_ratios, 1 - epsilon, 1 + epsilon) * advantages
        
        # (可选) SAMPO 核心改进：如果发现 Adv < 0 且极度劣质，直接 Mask 掉其 Loss
        loss = apply_sampo_masking(surrogate_loss, clipped_loss, advantages, is_ratios)
        policy_loss = -loss.mean()
        
        # 4. 反向传播，更新模型参数
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


    # ---------------------------------------------------------
    # 阶段 3：Diagnostics 阶段 (盘后分析，生成图 4 c, d)
    # ---------------------------------------------------------
    with torch.no_grad():
        # 1. 此时参数已更新，计算最新的新旧策略分布差异 (KL 散度)
        kl_divs = compute_kl_divergence(old_logits, new_logits) # 得到每个样本的拉扯步长
        
        # 初始化分类统计池
        group_kl_sums = {
            "Adv < 0 & IS < 1": 0.0,
            "Adv < 0 & IS > 1": 0.0,
            "Adv > 0 & IS < 1": 0.0,
            "Adv > 0 & IS > 1": 0.0
        }
        
        total_kl = kl_divs.sum()
        
        # 2. 数据分组 (Partitioning)
        # 根据固定的 Advantage 符号、当前的 IS 大小，对 batch 内的样本进行归类累加
        for i in range(batch_size):
            adv = advantages[i]
            is_ratio = is_ratios[i] # 更新后的最终 IS
            kl = kl_divs[i].sum()
            
            if adv < 0 and is_ratio < 1:
                group_kl_sums += kl
            elif adv < 0 and is_ratio >= 1:
                group_kl_sums += kl
            elif adv >= 0 and is_ratio < 1:
                group_kl_sums += kl
            elif adv >= 0 and is_ratio >= 1:
                group_kl_sums += kl
                
        # 3. 归一化处理 (Normalization)
        # 算出每一类糟糕样本或优质样本对总 KL 散度的“贡献占比”
        normalized_kl = {}
        for group_name, kl_sum in group_kl_sums.items():
            normalized_kl[group_name] = kl_sum / total_kl  # 这就是图 4 c,d 的纵轴高度！
            
        # 4. 将分类的归一化 KL 写入日志绘图
        # (原论文中还会加入熵值 entropy level 的分类维度)
        log_to_tensorboard("Normalized_KL_Distributions", normalized_kl)
        
    # 本轮更新彻底结束，将更新后的策略同步为下一轮的旧策略
    old_policy.load_state_dict(current_policy.state_dict())