import pandas as pd

from eval.ambiguity import calculate_ambiguity_for_df
from eval.answerability import compute_answerability_for_df
from eval.disclosure import compute_disclosure_for_df
from eval.distractors_quality import compute_distractor_quality_for_df
from eval.negation import starts_with_negation
from eval.originality import calculate_originality_for_df
from eval.question_check import is_question
from eval.readability import calculate_readability_for_df
from eval.relevance import calculate_relevance_for_df
from eval.difficulty import compute_difficulty_for_df

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import math


def eval_dataframe(df_merged: pd.DataFrame,
                   # openai params
                   openai_key: str,
                   temp=0.1,
                   max_completion_tokens=1,
                   output_file_path='./mcqs_eval.csv',
                   # answerability params
                   compute_answerability=True,
                   answerability_col='gpt_answer',
                   answerability_system_prompt=None,
                   # originality params
                   compute_originality=True,
                   originality_col='originality',
                   # readability params
                   compute_readability=True,
                   readability_col='readability',
                   # negation params
                   compute_negation=True,
                   negation_col='starts_with_negation',
                   # is_question params
                   compute_is_question=True,
                   is_question_col='is_question',
                   # relevance_params
                   compute_relevance=True,
                   relevance_col='relevance',
                   # ambiguity_params
                   compute_ambiguity=True,
                   ambiguity_col='ambiguity',
                   # disclosure params,
                   compute_disclosure=True,
                   disclosure_system_prompt=None,
                   disclosure_col='disclosure',
                   # difficulty params
                   compute_difficulty=True,
                   difficulty_system_prompt=None,
                   difficulty_col='difficulty',
                   # distractors quality params
                   compute_distractors_quality=True,
                   distractors_quality_system_prompt=None,
                   distractors_quality_col='distractors_quality',
                   # general df params
                   question_col='question',
                   option_a_col='option_a',
                   option_b_col='option_b',
                   option_c_col='option_c',
                   option_d_col='option_d',
                   correct_option_col='correct_option',
                   lisa_sheet_col='content_raw'):
    try:
        if compute_originality:
            df_merged = calculate_originality_for_df(df_merged,
                                                     originality_col=originality_col,
                                                     question_col=question_col,
                                                     lisa_sheet_col=lisa_sheet_col)

        if compute_readability:
            df_merged = calculate_readability_for_df(df_merged,
                                                     readability_col=readability_col,
                                                     question_col=question_col)

        if compute_negation:
            df_merged[negation_col] = df_merged[question_col].apply(starts_with_negation)

        if compute_is_question:
            df_merged[is_question_col] = df_merged[question_col].apply(is_question)

        if compute_relevance:
            df_merged = calculate_relevance_for_df(df_merged,
                                                   relevance_col=relevance_col,
                                                   question_col=question_col,
                                                   lisa_sheet_col=lisa_sheet_col)

        if compute_ambiguity:
            df_merged = calculate_ambiguity_for_df(df_merged,
                                                   correct_option_col=correct_option_col,
                                                   option_a_col=option_a_col,
                                                   option_b_col=option_b_col,
                                                   option_c_col=option_c_col,
                                                   option_d_col=option_d_col,
                                                   ambiguity_col=ambiguity_col)

        if compute_answerability and answerability_system_prompt is not None:
            df_merged = compute_answerability_for_df(df_merged,
                                                     api_key=openai_key,
                                                     question_col=question_col,
                                                     option_a_col=option_a_col,
                                                     option_b_col=option_b_col,
                                                     option_c_col=option_c_col,
                                                     option_d_col=option_d_col,
                                                     model_answer_col=answerability_col,
                                                     context_col=lisa_sheet_col,
                                                     system_prompt=answerability_system_prompt,
                                                     temp=temp,
                                                     max_completion_tokens=max_completion_tokens)

        if compute_disclosure and disclosure_system_prompt is not None:
            df_merged = compute_disclosure_for_df(df_merged,
                                                  api_key=openai_key,
                                                  question_col=question_col,
                                                  disclosure_col=disclosure_col,
                                                  system_prompt=disclosure_system_prompt,
                                                  temp=temp,
                                                  max_completion_tokens=max_completion_tokens)

        if compute_difficulty and difficulty_system_prompt is not None:
            df_merged = compute_difficulty_for_df(df_merged,
                                                  api_key=openai_key,
                                                  question_col=question_col,
                                                  difficulty_col=difficulty_col,
                                                  system_prompt=difficulty_system_prompt,
                                                  temp=temp,
                                                  max_completion_tokens=max_completion_tokens)

        if compute_distractors_quality and distractors_quality_system_prompt is not None:
            df_merged = compute_distractor_quality_for_df(df_merged,
                                                          api_key=openai_key,
                                                          question_col=question_col,
                                                          option_a_col=option_a_col,
                                                          option_b_col=option_b_col,
                                                          option_c_col=option_c_col,
                                                          option_d_col=option_d_col,
                                                          correct_option_col=correct_option_col,
                                                          distractor_quality_col=distractors_quality_col,
                                                          system_prompt=distractors_quality_system_prompt,
                                                          temp=temp,
                                                          max_completion_tokens=max_completion_tokens)
    except Exception as e:
        print(e)
    finally:
        return df_merged


def process_batch(batch_data):
    batch_merged_mcqs, kwargs = batch_data
    # Add batch_id to kwargs
    # Import eval_df here to avoid circular imports
    result = eval_dataframe(batch_merged_mcqs, **kwargs)
    return result


def eval_dataframe_parallel(df_mcqs: pd.DataFrame,
                            df_lisa_sheets: pd.DataFrame,
                            num_workers: int = 10,
                            lisa_sheet_id_col='id',
                            lisa_sheet_col='content_raw',
                            merge=False,
                            **kwargs):
    """
    Parallel version of eval_df that processes data in batches using multiple workers.
    
    Parameters:
    -----------
    df_mcqs : pd.DataFrame
        First input dataframe
    df_lisa_sheets : pd.DataFrame
        Second input dataframe with the same length as df_mcqs
    num_workers : int, default=12
        Number of parallel workers to use
    **kwargs : dict
        Additional parameters to pass to eval_df
        
    Returns:
    --------
    pd.DataFrame
        Merged and processed dataframe, sorted by index
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Ensure dataframes are the same length
    #if merge and len(df_mcqs) != len(df_lisa_sheets):
    #    raise ValueError(f"Input dataframes must have the same length. Got {len(df_mcqs)} and {len(df_lisa_sheets)}.")

    # Calculate batch size
    total_rows = len(df_mcqs)
    batch_size = math.ceil(total_rows / num_workers)

    df_merged = df_mcqs if not merge else pd.merge(df_mcqs,
                                                   df_lisa_sheets[[lisa_sheet_id_col, lisa_sheet_col]],
                                                   on=lisa_sheet_id_col, how='left')
    


    # Create batches
    batches = []
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        batches.append((
            df_merged.iloc[i:end_idx].copy(),
            kwargs
        ))

    logger.info(f"Created {len(batches)} batches with batch size of {batch_size}")

    # Process batches in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        futures = list(tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))

        # Collect results
        for future in futures:
            results.append(future)

    # Combine results
    logger.info("Combining results from all batches")
    if results:
        combined_df = pd.concat(results, ignore_index=False)
        # Sort by index
        combined_df = combined_df.sort_index()
        logger.info(f"Combined result has {len(combined_df)} rows")
        return combined_df
    else:
        logger.warning("No results returned from batches")
        return pd.DataFrame()
