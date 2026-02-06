# GISA: A Benchmark for General Information-Seeking Assistant

GISA is a benchmark for General Information-Seeking Assistants with 373 human-crafted queries that reflect real-world information needs. It includes both stable and live subsets, four structured answer formats (item, set, list, table), and complete human search trajectories for every query.

## Links
- LeaderBoard: (TBD)
- arXiv: (TBD)
- Dataset: https://huggingface.co/datasets/RUC-NLPIR/GISA

## Benchmark Highlights
- **Diverse answer formats with deterministic evaluation.**  
  GISA uses four structured answer types (item, set, list, table) with strict matching metrics for reproducible evaluation, avoiding subjective LLM judging while preserving task diversity.
- **Unified deep + wide search capabilities.**  
  Tasks require both vertical reasoning and horizontal information aggregation across sources, evaluating long-horizon exploration and summarization in one benchmark.
- **Dynamic, anti-static evaluation.**  
  Queries are split into stable and live subsets; the live subset is periodically updated to reduce memorization and keep the benchmark challenging over time.
- **Process-level supervision via human trajectories.**  
  Full human search trajectories are provided for every query, serving as gold references for process reward modeling and imitation learning while validating task solvability.

## Submission
Please follow our submission instructions (link coming soon) and open a pull request on the GitHub repository. We review PRs periodically and merge approved results.

## Citation
```bibtex
@inproceedings{gisa2026,
  title     = {GISA: A Benchmark for General Information Seeking Assistant},
  author    = {Yutao Zhu and Xingshuo Zhang and Maosen Zhang and Jiajie Jin and Liancheng Zhang and Xiaoshuai Song and Kangzhi Zhao and Wencong Zeng and Ruiming Tang and Han Li and Ji-Rong Wen and Zhicheng Dou},
  booktitle = {TBD},
  year      = {2026}
}
