import { environment } from '../../environments/environment';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AnswerabilityRequest {
  model_type: 'ollama' | 'hf';
  model_name: string;
  dataset_name: string;
  dataset_split: string;
  preprocess_mode: string;
  question_col: string;
  option_a_col: string;
  option_b_col: string;
  option_c_col: string;
  option_d_col: string;
  correct_col: string;
  sample_size: number | null;
}

export interface AnswerabilityJob {
  job_id: string;
  status: 'pending' | 'loading_dataset' | 'loading_model' | 'running' | 'done' | 'error' | 'cancelled';
  progress: number;
  total: number;
  accuracy: number | null;
  correct: number;
  error: string | null;
}

export interface AnswerabilityResult {
  job_id: string;
  model_name: string;
  model_type: string;
  dataset_name: string;
  sample_size: number;
  accuracy: number;
  correct: number;
  date: string;
}

export interface ModelCardMetadata {
  license: string | null;
  language: string[];
  pipeline_tag: string | null;
  downloads_last_month: number | null;
  likes: number | null;
  tags: string[];
}

export interface ModelCardInfo {
  model_id: string;
  metadata: ModelCardMetadata;
  sections: Record<string, string>;
  extractor_error: string | null;
}

@Injectable({ providedIn: 'root' })
export class AnswerabilityService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  getOllamaModels(): Observable<{ models: string[]; error?: string }> {
    return this.http.get<{ models: string[]; error?: string }>(
      `${this.apiUrl}/answerability/models/ollama`
    );
  }

  getOpenAICompatModels(): Observable<{ models: string[]; error?: string; base_url?: string }> {
    return this.http.get<{ models: string[]; error?: string; base_url?: string }>(
      `${this.apiUrl}/answerability/models/openai-compat`
    );
  }

  runEvaluation(req: AnswerabilityRequest): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${this.apiUrl}/answerability/run`, req);
  }

  getJobStatus(jobId: string): Observable<AnswerabilityJob> {
    return this.http.get<AnswerabilityJob>(`${this.apiUrl}/answerability/job/${jobId}`);
  }

  getResults(): Observable<{ results: AnswerabilityResult[] }> {
    return this.http.get<{ results: AnswerabilityResult[] }>(`${this.apiUrl}/answerability/results`);
  }

  cancelJob(jobId: string): Observable<{ job_id: string; message: string }> {
    return this.http.delete<{ job_id: string; message: string }>(
      `${this.apiUrl}/answerability/job/${jobId}`
    );
  }

  getModelCard(modelId: string): Observable<ModelCardInfo> {
    return this.http.get<ModelCardInfo>(
      `${this.apiUrl}/model-card/${encodeURIComponent(modelId)}`
    );
  }
}
