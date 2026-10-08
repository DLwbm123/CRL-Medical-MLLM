# P8 全覆盖输出协议审计

覆盖18轨迹×(初始16+最终16 probe+before16+after16)，共1152输出记录；其中576为probe记录。全覆盖表见 p8_format_cases.csv。重复初始预测是共享缓存，不是576个独立病例。使用原P3锁定分类与 marked_choice_v1_eval_only，不新增逐病例规则。原严格结果不变。

| 轮次 | arm | 输出 | cursor | 记录数 | 原解析数 | 原正确数 | 额外解析数 | 额外正确数 | 原无效分类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P5 | m | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P5 | m | probe | 16 | 48 | 39 | 10 | 40 | 10 | {"explicit_choice_unsupported_syntax": 3, "normal_end_missing_final_marker_or_explicit_choice": 4, "prose_final_payload_without_unambiguous_identifier": 1, "token_cap_without_parseable_final_choice": 1} |
| P5 | m | before | all | 48 | 39 | 11 | 40 | 11 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 6, "token_cap_without_parseable_final_choice": 1} |
| P5 | m | after | all | 48 | 38 | 9 | 38 | 9 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 6, "token_cap_without_parseable_final_choice": 2} |
| P5 | v | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P5 | v | probe | 16 | 48 | 40 | 8 | 40 | 8 | {"explicit_choice_unsupported_syntax": 3, "normal_end_missing_final_marker_or_explicit_choice": 3, "token_cap_without_parseable_final_choice": 2} |
| P5 | v | before | all | 48 | 35 | 10 | 36 | 10 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 10, "token_cap_without_parseable_final_choice": 1} |
| P5 | v | after | all | 48 | 36 | 8 | 37 | 8 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 7, "token_cap_without_parseable_final_choice": 3} |
| P6 | m | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P6 | m | probe | 16 | 48 | 36 | 6 | 40 | 8 | {"explicit_choice_unsupported_syntax": 4, "normal_end_missing_final_marker_or_explicit_choice": 5, "token_cap_without_parseable_final_choice": 3} |
| P6 | m | before | all | 48 | 35 | 10 | 36 | 10 | {"explicit_choice_unsupported_syntax": 4, "normal_end_missing_final_marker_or_explicit_choice": 8, "token_cap_without_parseable_final_choice": 1} |
| P6 | m | after | all | 48 | 38 | 12 | 39 | 12 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 8} |
| P6 | v | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P6 | v | probe | 16 | 48 | 37 | 6 | 40 | 6 | {"explicit_choice_unsupported_syntax": 4, "normal_end_missing_final_marker_or_explicit_choice": 5, "token_cap_without_parseable_final_choice": 2} |
| P6 | v | before | all | 48 | 40 | 8 | 42 | 8 | {"explicit_choice_unsupported_syntax": 4, "normal_end_missing_final_marker_or_explicit_choice": 4} |
| P6 | v | after | all | 48 | 38 | 11 | 41 | 11 | {"explicit_choice_unsupported_syntax": 5, "normal_end_missing_final_marker_or_explicit_choice": 4, "token_cap_without_parseable_final_choice": 1} |
| P7 | m | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P7 | m | probe | 16 | 48 | 41 | 11 | 42 | 11 | {"explicit_choice_unsupported_syntax": 4, "normal_end_missing_final_marker_or_explicit_choice": 2, "token_cap_without_parseable_final_choice": 1} |
| P7 | m | before | all | 48 | 35 | 11 | 36 | 11 | {"explicit_choice_unsupported_syntax": 3, "normal_end_missing_final_marker_or_explicit_choice": 10} |
| P7 | m | after | all | 48 | 39 | 11 | 40 | 11 | {"explicit_choice_unsupported_syntax": 2, "normal_end_missing_final_marker_or_explicit_choice": 7} |
| P7 | v | probe | 0 | 48 | 39 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 6, "normal_end_missing_final_marker_or_explicit_choice": 3} |
| P7 | v | probe | 16 | 48 | 35 | 9 | 39 | 9 | {"explicit_choice_unsupported_syntax": 5, "normal_end_missing_final_marker_or_explicit_choice": 5, "token_cap_without_parseable_final_choice": 3} |
| P7 | v | before | all | 48 | 40 | 10 | 40 | 10 | {"explicit_choice_unsupported_syntax": 1, "normal_end_missing_final_marker_or_explicit_choice": 6, "token_cap_without_parseable_final_choice": 1} |
| P7 | v | after | all | 48 | 37 | 12 | 39 | 12 | {"explicit_choice_unsupported_syntax": 3, "normal_end_missing_final_marker_or_explicit_choice": 5, "token_cap_without_parseable_final_choice": 3} |

## 全部原正确 probe 损害

| 轮次 | seed | arm | 组 | 原解析 | 额外语法恢复正确 | 分类 | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P5 | 49 | v | P06 | False | False | normal_end_missing_final_marker_or_explicit_choice | 134 |
| P5 | 49 | v | P09 | False | False | token_cap_without_parseable_final_choice | 2048 |
| P5 | 50 | m | P09 | True | False | parsed_control | 219 |
| P6 | 51 | v | P09 | False | False | token_cap_without_parseable_final_choice | 2048 |
| P6 | 52 | m | P06 | False | True | explicit_choice_unsupported_syntax | 116 |
| P6 | 52 | v | P06 | True | False | parsed_control | 155 |
| P6 | 52 | v | P09 | False | False | token_cap_without_parseable_final_choice | 2048 |
| P6 | 53 | m | P06 | False | True | explicit_choice_unsupported_syntax | 202 |
| P6 | 53 | m | P09 | False | False | token_cap_without_parseable_final_choice | 2048 |
| P7 | 55 | v | P09 | False | False | token_cap_without_parseable_final_choice | 2048 |

共10次原正确probe损害，落在2个不同已知组。只有2次由统一额外语法解析明确恢复；其余包括 token上限、缺少答案和已解析错误。候选臂的6次损害没有额外语法正确恢复。

P7 v55 的原探针B对应 P09：2048 token达到上限，原解析与统一额外解析均不能识别合法最终选项。这说明没有形成这两个锁定协议可用的答案；并非已证明医学知识遗忘。额外规则以外的临床语义意图未作人工判定，不能断言文本中绝无其他表达。没有借助真值猜测临床段落或隐含答案。

P09在多个seed/方法重复失效，P06既出现纯语法恢复，也出现正常结束缺少答案和解析错误。相同病例重复不能写成新的独立证据。语法标签不自动代表可恢复的正确答案；表中 extra_correct 为独立评分结果。人工临床语义意图均标记未判定。
