import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { MCQCard } from '../models/mcq.model';

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
   * Get list of assigned MCQ IDs for current user
   */
  getAssignedMcqs(): Observable<{ mcq_ids: string[], total: number, user: string }> {
    return this.http.get<{ mcq_ids: string[], total: number, user: string }>(
      `${this.apiUrl}/mcq/assigned`
    );
  }

  /**
   * Get a specific MCQ by ID
   */
  getMcqById(mcqId: string): Observable<MCQCard> {
    return this.http.get<MCQCard>(`${this.apiUrl}/mcq/${mcqId}`);
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
  getAvailableModels(): Observable<{ model: string, count: number }[]> {
    return this.http.get<{ model: string, count: number }[]>(`${this.apiUrl}/mcq-models`);
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
   * Save current progress (for pause/resume)
   */
  saveProgress(data: {
    current_index: number,
    mcq_list: string[],
    current_mcq_state?: any
  }): void {
    localStorage.setItem('mcq_evaluation_progress', JSON.stringify(data));
  }

  /**
   * Load saved progress
   */
  loadProgress(): {
    current_index: number,
    mcq_list: string[],
    current_mcq_state?: any
  } | null {
    const saved = localStorage.getItem('mcq_evaluation_progress');
    return saved ? JSON.parse(saved) : null;
  }

  /**
   * Clear saved progress
   */
  clearProgress(): void {
    localStorage.removeItem('mcq_evaluation_progress');
  }
}
