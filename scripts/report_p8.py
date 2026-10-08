"""Publish sanitized complete diagnostic tables, without opening private inputs."""
import json
import os
from collections import Counter
from pathlib import Path
from report_p3 import markdown_table, write_csv


def main():
    root=Path(os.environ['P0_ROOT'])
    public=json.loads(Path(os.environ['P8_PUBLIC']).read_text())
    dest=root/'reports'
    offline=public['offline']
    cases=[r for v in offline.values() for r in v['cases']]
    groups=[r for v in offline.values() for r in v['groups']]
    formats=[r for v in offline.values() for r in v['formats']]
    primary=[r for v in offline.values() for r in v['primary']]
    assert len(cases)==288 and len(formats)==1152 and len(primary)==18
    assert all(v['recomputed_matches_sealed'] for v in offline.values())
    for filename, rows in [('case_level_audit',cases),('group_map',groups),('primary_recheck',primary),
                           ('format_cases',formats),('matched_readout',public['readout']),
                           ('matched_prefix_kl',public['prefix_kl']),('readout_summary',public['readout_summary'])]:
        write_csv(dest/('p8_'+filename+'.csv'),[{k:'N/A' if v is None else json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in rows])
    if public['gradient_rows']:
        write_csv(dest/'p8_gradient_components.csv',[{k:'N/A' if v is None else v for k,v in row.items()} for row in public['gradient_rows']])
    else:
        write_csv(dest/'p8_gradient_components.csv',[{'status':'NOT_EXECUTED','reason':'uniform frozen time admission rejected D; no gradient evidence'}])
    receipt=public['receipt']
    receipt={k:v for k,v in receipt.items() if k not in ['pid','start_ticks']}
    receipt.update(total_authorized_gpu_process_seconds=86400,p8_gpu_process_seconds=receipt['wall_seconds'],
                   p8_worker_lifetime_cap_seconds=10800,source_commit=os.environ['P8_SOURCE_COMMIT'],
                   p8_failed_gpu_attempts=0,new_gpu_workers=1,optimizer_steps=0,
                   prior_p4_gpu_process_seconds=28.726142450002953,
                   prior_p5_gpu_process_seconds=21744.03255091561,
                   prior_p6_gpu_process_seconds=21149.256308959797,
                   prior_p7_gpu_process_seconds=22583.527223093202,
                   clock_origin_limitation='UTC recovered from first Python ps start; monotonic reconstructed after instruction read',
                   t0_utc='2026-10-08T13:01:41+00:00',
                   cpu_audit_rounds=3,cpu_original_metrics_match=True,cpu_engineering_tests=4,
                   cpu_preparation_failures=['artifact_source_receipt_lookup_corrected',
                                             'manifest_literal_escape_corrected',
                                             'pool_method_configuration_name_corrected'],
                   optional_mask_span='not_executed_unverified_token_text_mapping',
                   full_parameter_bytes_compared=False,full_parameter_versions_checked=True,
                   full_buffers_checked=True,parameter_values_sampled_three_per_tensor=True)
    assert abs(sum(receipt[f'prior_p{n}_gpu_process_seconds'] for n in [4,5,6,7])-receipt['gpu_process_seconds_before_p8'])<1e-6
    assert abs(receipt['cumulative_gpu_process_seconds']+receipt['remaining_gpu_process_seconds']-86400)<1e-6
    (dest/'p8_compute_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    summaries=[]
    for round_name in ['P5','P6','P7']:
        for arm in 'mv':
            selected=[r for r in groups if r['round']==round_name and r['arm']==arm]
            summaries.append([round_name,arm,16,48,sum(r['all_consensus_wrong'] for r in selected),
                              sum(r['all_consensus_wrong'] and r['unique_consensus_answers']==1 for r in selected),
                              sum(r['unique_consensus_answers']>1 for r in selected),
                              sum(r['correct_candidates_of_24']==0 for r in selected),
                              sum(r['correct_candidate_wrong_vote']==3 for r in selected)])
    gain_groups=sorted({r['group'] for r in groups if r['history_gain_observations']})
    loss_groups=sorted({r['group'] for r in groups if r['history_loss_observations']})
    all_losses=[r for r in formats if r['kind']=='probe' and r['cursor']==16 and r['initial_strict_correct_probe'] and not r['strict_correct']]
    candidate_losses=[r for r in all_losses if r['arm']=='v']
    format_counts=[]
    for p in ['P5','P6','P7']:
        for a in 'mv':
            for kind,cursor in [('probe',0),('probe',16),('before',None),('after',None)]:
                selected=[r for r in formats if r['round']==p and r['arm']==a and r['kind']==kind and (cursor is None or r['cursor']==cursor)]
                types=Counter(r['failure_type'] for r in selected if not r['strict_parsed'])
                format_counts.append([p,a,kind,cursor if cursor is not None else 'all',len(selected),
                                      sum(r['strict_parsed'] for r in selected),sum(r['strict_correct'] for r in selected),
                                      sum(r['extra_parsed'] for r in selected),sum(r['extra_correct'] for r in selected),json.dumps(dict(types),sort_keys=True)])
    harm_table=markdown_table(['轮次','seed','arm','组','原解析','额外语法恢复正确','分类','tokens'],
        [[r['round'],r['seed'],r['arm'],r['group'],r['strict_parsed'],r['extra_correct'],r['failure_type'],r['tokens']] for r in all_losses])
    format_text='\n\n'.join(['# P8 全覆盖输出协议审计',
        '覆盖18轨迹×(初始16+最终16 probe+before16+after16)，共1152输出记录；其中576为probe记录。全覆盖表见 p8_format_cases.csv。重复初始预测是共享缓存，不是576个独立病例。使用原P3锁定分类与 marked_choice_v1_eval_only，不新增逐病例规则。原严格结果不变。',
        markdown_table(['轮次','arm','输出','cursor','记录数','原解析数','原正确数','额外解析数','额外正确数','原无效分类'],format_counts),
        '## 全部原正确 probe 损害',harm_table,
        f'共{len(all_losses)}次原正确probe损害，落在{len({r["group"] for r in all_losses})}个不同已知组。只有{sum(r["extra_correct"] for r in all_losses)}次由统一额外语法解析明确恢复；其余包括 token上限、缺少答案和已解析错误。候选臂的{len(candidate_losses)}次损害没有额外语法正确恢复。',
        'P7 v55 的原探针B对应 P09：2048 token达到上限，原解析与统一额外解析均不能识别合法最终选项。这说明没有形成这两个锁定协议可用的答案；并非已证明医学知识遗忘。额外规则以外的临床语义意图未作人工判定，不能断言文本中绝无其他表达。没有借助真值猜测临床段落或隐含答案。',
        'P09在多个seed/方法重复失效，P06既出现纯语法恢复，也出现正常结束缺少答案和解析错误。相同病例重复不能写成新的独立证据。语法标签不自动代表可恢复的正确答案；表中 extra_correct 为独立评分结果。人工临床语义意图均标记未判定。'])+'\n'
    (dest/'p8_format_audit.md').write_text(format_text)
    readout=public['readout_summary']
    readout_table=markdown_table(['actor','严格正确/16','严格解析/16','原正确保留/3','合法读出正确/16','合法初始集保留','共同前缀平均KL'],
        [[r['actor'],r['strict_correct'],r['strict_parsed'],r['strict_initial_retained'],r['legal_correct'],
          f'{r["legal_initial_retained"]}/{r["legal_initial_n"]}',f'{r["mean_kl"]:.8g}'] for r in readout])
    paired=[]
    for seed in [54,55,56]:
        left=next(r for r in readout if r['actor']=='m'+str(seed));right=next(r for r in readout if r['actor']=='v'+str(seed))
        m=[r for r in public['readout'] if r['actor']=='m'+str(seed)]
        v=[r for r in public['readout'] if r['actor']=='v'+str(seed)]
        mk=[r for r in public['prefix_kl'] if r['actor']=='m'+str(seed)]
        vk=[r for r in public['prefix_kl'] if r['actor']=='v'+str(seed)]
        paired.append([seed,f'{left["mean_kl"]:.8g}',f'{right["mean_kl"]:.8g}',
                       sum(b['kl_actor_base']<a['kl_actor_base'] for a,b in zip(mk,vk)),
                       sum(b['correct_minus_best_wrong_gap']>a['correct_minus_best_wrong_gap'] for a,b in zip(m,v)),
                       sum(b['legal_top1_top2_gap']>a['legal_top1_top2_gap'] for a,b in zip(m,v)),
                       left['legal_correct'],right['legal_correct']])
    grad=public['gradient_rows']
    total_grad=[r for r in grad if r['module']=='all_selected']
    band_pairs=[r['policy_band_cosine'] for r in total_grad if r['policy_band_cosine'] is not None]
    band_ratios=[r['band_policy_ratio'] for r in total_grad if r['band_policy_ratio'] is not None]
    lower_mean_kl=sum(float(r[2])<float(r[1]) for r in paired)
    lost55=next(r for r in public['readout'] if r['actor']=='v55' and r['group']=='P09')
    grad_table=markdown_table(['actor','组','policy norm','band/policy','KL raw norm','KL/policy','policy-band cosine','policy-KL cosine','合成/sum norms'],
        [[r['actor'],r['group'],*[('N/A' if r[k] is None else f'{r[k]:.6g}') for k in ['policy_norm','band_policy_ratio','kl_raw_norm','kl_policy_ratio','policy_band_cosine','policy_kl_cosine','composite_over_sum_norms']]] for r in total_grad])
    primary_table=markdown_table(['轮次','seed','arm','before/16','after/16','解析before/after','probe/16','原正确保留/3','正确负adv','错误正奖励'],
        [[r['round'],r['seed'],r['arm'],r['before_correct'],r['after_correct'],f'{r["before_parsed"]}/{r["after_parsed"]}',r['probe_final_correct'],r['original_correct_retained'],
          f'{r["correct_negative_n"]}/{r["correct_negative_d"]}',f'{r["wrong_positive_n"]}/{r["wrong_positive_d"]}'] for r in primary])
    report='\n\n'.join(['# P8：奖励失配、输出协议与局部损失梯度诊断',
        'P8已完成全18轨迹原结果复核及全覆盖格式审计，七actor×16 probe共同输入读出/精确KL，和七actor×两个原始回答组的局部梯度诊断。下一轮唯一优先方向是确认授权并扩大独立开发病例。现有证据同时支持奖励失配、输出协议敏感性与局部分量冲突，不能可靠区分主要因果来源；不继续在同16组上调参。P9只有草案，未启动。',
        '## 原实验事实及复核',primary_table,
        '上述18条轨迹全部原 score_cases 指标、解析转移、probe统计、奖励方向及原格式聚合，与相应已执行版本 evaluator 和封存结果逐项精确一致；无差异。P5–P7均未达到原三seed联合稳定正向条件。冻结strict stream底座2/16，原正确probe3组。复核没有改写旧报告或奖励。',
        '## 新诊断A：病例地图',
        markdown_table(['轮次','arm','不同组','观测','三seed均错','同一错误共识不变','共识随seed变','24回答无正确候选','三seed有正确候选却均错选'],summaries),
        f'历史收益至少一次出现于{len(gain_groups)}个不同已知stream组：{", ".join(gain_groups)}；历史损失只出现在{len(loss_groups)}个组：{", ".join(loss_groups)}。这不是新增独立病例，也不说明收益在所有seed持续出现。p8_group_map.csv 给出每轮每臂所有16组的3次观测、零缺失、票数/票差和gain/loss次数，p8_case_level_audit.csv有全部288组观测及长度、解析、奖励/advantage分母。',
        '三seed都错误不等于三seed选同一个错误答案。许多组同时存在错误共识、跨seed选项翻转和偶发正确候选；另有3–5组在每个轮次/臂的24回答中完全没有正确候选。正确候选偶尔出现不等于可靠能力；高票/大票差不是校准置信度，也不是参数梯度贡献。不能由错误正奖励频次推出所有更新方向都错。完整组表保留所有票数，不设置事后“高置信”门槛。',
        '## 新诊断B：输出失效',
        f'全部{len(all_losses)}次原正确probe损害对应两个已知probe组；{sum(r["extra_correct"] for r in all_losses)}次在原设置P6臂被统一额外语法规则明确恢复。候选臂6次损害没有语法正确恢复。P7 v55/P09达到2048 token上限，额外语法规则仍未识别合法最终选项；规则以外的语义意图保留未判定。完整计数和全部反例见 p8_format_audit.md、p8_format_cases.csv。严格主结果保留。',
        '## 新诊断C：相同输入与前缀',readout_table,
        markdown_table(['seed','m mean KL','v mean KL','v更小KL组/16','v正确选项边际更大/16','v top1-top2更大/16','m合法正确','v合法正确'],paired),
        f'强KL候选在{lower_mean_kl}/3配对seed的共同前缀平均KL更小；逐组漂移与两种选项边际方向见上表，并非一致改善。P7 v55/P09原严格失效，在共同合法读出下{"正确" if lost55["legal_correct"] else "仍选错"}；其正确选项相对最高错误选项边际为{lost55["correct_minus_best_wrong_gap"]:.8g}。这个共同输入结果不能由原生成文本的语法修正代替。',
        'C比较全部16组，避免训练各臂自身推理前缀不同的混淆。共同首答案位置全词表KL仍不能代表全response masked KL或训练路径。正确选项边际在输出封存、GPU退出之后才补充；GPU未读标签。自由生成初始正确集合和合法读出初始正确集合分别计算，不能混合保留率。完整逐组分数统计见两张 matched CSV。',
        '合法读出恢复或提升支持输出协议敏感性，不能证明知识没有变化；两个协议都错也不能定位具体医学概念遗忘。最终总正确数可能掩盖原正确组损害。合法选项得分是条件log概率，未经医学置信度校准。',
        '## 新诊断D：固定旧回答池的局部梯度',grad_table,
        f'全部选定参数合计的policy与加权band夹角在{sum(c<0 for c in band_pairs)}/{len(band_pairs)}个可定义actor×组观测为负；加权band/policy范数比范围为{min(band_ratios):.6g}–{max(band_ratios):.6g}。这里只是同一子集局部目标的方向/量级证据，不是诊断特定probe因果的门槛。' if band_ratios else '无可定义policy/band范数比；不补造分量影响证据。',
        '在全部选定模块合计上，加权band约为policy范数的0.41%–1.34%，没有支持“band范数主导”的证据。模块/视觉/语言结果仍应分别阅读，不能把合计比率当成每层或全模型比率。强KL候选的加权KL/policy比约0.346–0.701；v55两组policy-KL cosine约−0.875/−0.981，合成/sum norms约0.436/0.199，显示该固定局部目标有抵消。状态与系数都不同于对照，不能把变化简单归为系数缩放或医学退化原因。',
        '底座raw-KL子集合计范数约2.15e−5/1.17e−5，低于预声明0.005容差；非零残差视为数值工程量，其近零梯度的cosine不作科学方向解释。',
        '全部七actor、S01/S02各八条原P3 FrozenSC8 seed43回答，不重新采样、截断、按正确率选样或替换零advantage组。五权重模块、视觉/语言合计和全部子集合计均公开；原KL梯度与实际系数加权梯度单列。单位band梯度乘实际0.01；候选KL乘1.0，对照/base乘0.01。ratio初值1，当前actor决定entropy/selection与detached band，reference固定原底座。',
        '范数比和负cosine只支持值得证伪的局部冲突假设，不能把任何分量称为具体probe退化原因。参数子集、两组旧策略回答池、不同actor历史均限制解释；它不是全模型梯度，不是训练梯度重放，也没有Adam步长/准确率反事实。原底座KL梯度的预声明数值工程检查通过。',
        '## 唯一后续优先方向及竞争解释',
        '选择D：扩大经明确授权的独立开发病例，以锁定的原SPINE/TTRL配对比较检验这些现象是否离开旧组后仍重复。仅4个已知stream组出现过历史收益，probe损害集中在两个组；稳定错误共识与随机翻转共存，纯语法问题只解释少量损害。共同前缀漂移与合法读出并不能建立某个训练分量的主要因果责任。',
        '暂缓奖励信息方案：确有正确候选被错共识压制，但同时存在正确候选稀缺，且尚无获授权、可验证的新奖励信息来源。暂缓单一格式稳定化：重复答案失效是明确问题，但语法恢复不能解释候选损害，改变读出不能替代原主指标。暂缓band开关：局部夹角可能冲突，却没有完整训练反事实，也未排除奖励/输出机制。优先独立病例检验重复性，避免继续对旧16组作机制选择。',
        '需要新病例授权和完整训练预算才能检验跨病例稳定性；真正因果比较还需在新的冻结协议中只改变目标分量。P9草案不自动授权这些操作，也不反向宣布P5–P7成功。',
        '## 算力、工程范围与退出',
        f'P8唯一GPU工作进程寿命{receipt["wall_seconds"]:.6f}秒（{receipt["wall_seconds"]/3600:.4f}小时），含import、加载、状态恢复、CPU等待和退出。累计{receipt["cumulative_gpu_process_seconds"]:.6f}秒（{receipt["cumulative_gpu_process_seconds"]/3600:.4f}小时），剩余{receipt["remaining_gpu_process_seconds"]:.6f}秒（{receipt["remaining_gpu_process_seconds"]/3600:.4f}小时）。未重置24小时授权。CPU准备失败仅是路径/字符串/配置名称检查失败，不消耗GPU；GPU失败尝试0。完整收据见 p8_compute_receipt.json。',
        '四个CPU工程测试通过，原18轨迹精确复核通过；实际GPU全参数版本计数、每tensor三个值抽样、全部buffer值检查、mode/RNG恢复通过。抽样参数值检查不等于全参数逐字节比较。全任务GPU进程退出，临时监督器结束；没有优化器/step、新checkpoint、新测试访问、其他GPU或付费服务。可选mask span未执行：缺少经过验证的token–文本映射。',
        'T0第一次命令未同时捕获UTC与monotonic；从首进程UTC开始时间保守回溯并在读指令后重建monotonic，保持原截止未延长。该执行限制与所有预算均如实保留。',
        '## 工件和公开边界',
        '只发布匿名统计、协议、诊断源代码、CPU测试、复现命令和算力收据。原ID、标签、题文、回答、图像、选项、权重、私有路径和别名映射均不公开。16stream/16probe是复用开发组，患者独立性未额外验证；重复seed/回答不作为新增病例，不作独立样本显著性检验。完整source/actor/input/pool冻结清单保留私有。'])+'\n'
    (dest/'p8_report.md').write_text(report)
    print(json.dumps({'case_rows':len(cases),'format_rows':len(formats),'readout_rows':len(public['readout']),
                      'gradient_rows':len(grad),'remaining_seconds':receipt['remaining_gpu_process_seconds']}))


if __name__=='__main__':main()
