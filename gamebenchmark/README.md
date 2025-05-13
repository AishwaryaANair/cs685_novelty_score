# Game Building Task

### Instructions to run the code

1. Please use colab to run nlp_data_test.ipynb. Ensure the runtime is GPU
2. Please load the dataset from the ./dataset folder in the colab environment via upload
3. Running all the codeblocks sequentially can reproduce the result as mentioned in the report


The results are in the results folder. The following is a description of the files

1. human_novelty_scores.csv - Obtained by running the colab notebook, it is a collection of all games with their corresponding novelty scores
2. llm_novelty_* - For each formula, we have created a separate result sheet
3. novelty_formula_evaluation_summary - Collective results of the evaluation present here

Additionally, in the games_data.xlsx, a comprehensive error analysis of the dataset, along with other information is present.

1. games_* - dataset used, as present in dataset/*
2. test_for_hyper_param_tuning - validation results on 10 randomly picked prompts versus the ground truth results and error analysis
3. human_novelty-score - same as the one present in results
4. response_llm - response of ground truth values, color is added to avoid confusion
5. response_llm_reorganized - same responses, just in a more processable format
6. llm_error_comparison - Error analysis of our benchmark
7. error_summary - same as the one present in the results section