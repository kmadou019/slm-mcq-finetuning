import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { MCQCard } from '../models/mcq.model';

export interface AssignedMCQ {
  mcq_id: string;
  model: string;
}

/**
 * MCQ Service - Handles all MCQ-related API calls
 */
@Injectable({
  providedIn: 'root'
})
export class McqService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api';

  /**
   * Get list of assigned MCQs with their models
   */
  getAssignedMcqs(): Observable<{ assignments: AssignedMCQ[], total: number, user: string }> {
    return this.http.get<{ assignments: AssignedMCQ[], total: number, user: string }>(
      `${this.apiUrl}/mcq/assigned`
    );
  }

  /**
   * Get a specific MCQ by ID and model
   */
  getMcqById(mcqId: string, model: string): Observable<MCQCard> {
    return this.http.get<MCQCard>(`${this.apiUrl}/mcq/${mcqId}?model=${encodeURIComponent(model)}`);
  }

  /**
   * Get a batch of MCQs
   */
  getMcqBatch(start: number = 0, limit: number = 10): Observable<{
    cards: MCQCard[],
    start: number,
    end: number,
    total: number,
    has_more: boolean
  }> {
    return this.http.get<any>(`${this.apiUrl}/mcq/batch?start=${start}&limit=${limit}`);
  }

  /**
   * Get available models with their MCQ counts
   */
  getAvailableModels(): Observable<{ model: string, count: number, available: number }[]> {
    return this.http.get<{ model: string, count: number, available: number }[]>(`${this.apiUrl}/mcq-models`);
  }

  /**
   * Submit validation for an MCQ
   */
  validateMcq(mcqId: string, validationData: {
    section_a_checks: any[],
    section_b_checks: any[],
    human_decision: string,
    human_feedback: string
  }): Observable<{ status: string, message: string, data: any }> {
    return this.http.post<any>(
      `${this.apiUrl}/mcq/${mcqId}/validate`,
      validationData
    );
  }

  /**
   * Lancer la génération de MCQs depuis un contenu custom (mode évaluateur)
   */
  startGeneration(content: string, modelSaveName: string, promptTemplate?: string): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${this.apiUrl}/generate`, {
      content,
      model_save_name: modelSaveName,
      prompt_template: promptTemplate
    });
  }

  /**
   * Extraire le texte d'un PDF côté serveur
   */
  extractPdfText(file: File): Observable<{ text: string }> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<{ text: string }>(`${this.apiUrl}/extract-pdf`, formData);
  }

  /**
   * Interroger l'état d'un job de génération (polling)
   */
  getGenerationStatus(jobId: string): Observable<{
    job_id: string;
    status: 'pending' | 'running' | 'done' | 'error';
    mcq_count: number;
    error: string | null;
  }> {
    return this.http.get<any>(`${this.apiUrl}/generate/${jobId}`);
  }

  /**
   * Cle localStorage dynamique par utilisateur
   */
  private getProgressKey(): string {
    const userJson = sessionStorage.getItem('mcq_current_user');
    if (userJson) {
      try {
        const user = JSON.parse(userJson);
        if (user.username) return `mcq_evaluation_progress_${user.username}`;
      } catch {}
    }
    return 'mcq_evaluation_progress_anonymous';
  }

  /**
   * Save current progress (for pause/resume)
   */
  saveProgress(data: {
    current_index: number,
    mcq_list: AssignedMCQ[],
    current_mcq_state?: any
  }): void {
    localStorage.setItem(this.getProgressKey(), JSON.stringify(data));
  }

  /**
   * Load saved progress
   */
  loadProgress(): {
    current_index: number,
    mcq_list: AssignedMCQ[],
    current_mcq_state?: any
  } | null {
    const saved = localStorage.getItem(this.getProgressKey());
    return saved ? JSON.parse(saved) : null;
  }

  /**
   * Clear saved progress
   */
  clearProgress(): void {
    localStorage.removeItem(this.getProgressKey());
  }
}
