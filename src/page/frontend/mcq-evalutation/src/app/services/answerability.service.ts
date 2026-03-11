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

@Injectable({ providedIn: 'root' })
export class AnswerabilityService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api';

  getOllamaModels(): Observable<{ models: string[]; error?: string }> {
    return this.http.get<{ models: string[]; error?: string }>(
      `${this.apiUrl}/answerability/models/ollama`
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
}
