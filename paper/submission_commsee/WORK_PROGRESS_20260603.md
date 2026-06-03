# Communications Earth & Environment Submission Progress

保存时间：2026-06-03

## 当前投稿状态

稿件已提交到 Nature Publishing Group / Communications Earth & Environment 的 Manuscript Tracking System，并已收到系统确认邮件。

稿件题目已统一为：

> Reproducible model-based AI planning for county-scale farmland consolidation in fragmented mountain landscapes

文章类型已统一为：

> Article

## 最终投稿文件

以下 3 个 PDF 是当前整理好的最终上传/投稿文件：

- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\00_cover_letter.pdf`
- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\01_main_manuscript.pdf`
- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\02_supplementary_information.pdf`

最新检查到的文件信息：

| File | Size | Last modified |
| --- | ---: | --- |
| `00_cover_letter.pdf` | 57,887 bytes | 2026-06-02 17:32:35 |
| `01_main_manuscript.pdf` | 925,514 bytes | 2026-06-02 17:32:36 |
| `02_supplementary_information.pdf` | 299,698 bytes | 2026-06-02 18:26:56 |

## 已完成的主要修改

- `source_main_codex.tex`
  - 题目统一为当前投稿题目。
  - 正文结构调整为 `Introduction / Results / Discussion / Methods`。
  - 摘要压缩到约 140 words。
  - 正文整体压缩，减少冗长表述。
  - 表格做了尺寸调整，避免明显 overfull box。

- `source_cover_letter.tex`
  - 题目统一为当前投稿题目。
  - 文章类型统一为 `Article`。

- `source_supplementary_codex.tex`
  - Supplementary Information 标题已修正。
  - 当前标题为：
    `Supplementary Information for "Reproducible model-based AI planning for county-scale farmland consolidation in fragmented mountain landscapes"`

- `README.md`
  - 已改写为投稿清单和首次投稿操作提示。

## 源文件位置

主要源文件：

- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\source_main_codex.tex`
- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\source_supplementary_codex.tex`
- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\source_cover_letter.tex`
- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\README.md`

## Nature / MTS 邮件状态

已收到 Communications Earth & Environment 的投稿确认邮件。邮件主要说明：

- 投稿已进入系统。
- 期刊是 open access，接收后会涉及 APC。
- 期刊使用 transparent peer review。
- 期刊强调 FAIR data 和数据归档要求。
- 接收发表时，支持结论的数据需要尽可能放入公开仓库。

另一封关于 data / code / materials availability 的邮件属于 Nature Portfolio 的常规政策提醒。若已经能满足数据归档政策，不需要立即操作；但建议主动准备一段数据可得性说明，尤其是原始地籍/三调数据不能公开再分发的部分。

建议后续可用的说明思路：

> Raw cadastral and Third National Land Survey records are subject to government data access restrictions and cannot be redistributed publicly. Public derived/anonymised data, the synthetic benchmark, the Buchanan case, code, model outputs, logs, and reproduction instructions are provided in the repository to support reproducibility.

## 关机后下一步

1. 登录 MTS，检查 manuscript status。
2. 确认 GitHub / anonymous repository / data repository 是否公开可访问。
3. 决定是否给编辑部补发一封简短邮件，说明受限原始数据和可公开复现实验材料的安排。
4. 准备 APC / open access funding 信息，等接收后再处理付款或机构协议。
5. 等待编辑部初筛或外审决定。

## Git 提交说明

本次任务使用临时 `safe.directory` 参数检查仓库状态，没有修改全局 Git 配置。
投稿包相关文件将随本记录一起提交到对应的 GitHub 仓库。
