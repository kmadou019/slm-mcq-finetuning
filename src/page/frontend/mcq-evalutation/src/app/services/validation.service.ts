import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { tap, catchError } from 'rxjs/operators';

/**
 * Service pour gerer les validations
 * Synchronise avec le backend et utilise localStorage comme cache
 * Cle composite mcq_id::model pour distinguer les MCQs de modeles differents
 */
@Injectable({
  providedIn: 'root'
})
export class ValidationService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api';

  private getStorageKey(): string {
    const userJson = sessionStorage.getItem('mcq_current_user');
    if (userJson) {
      try {
        const user = JSON.parse(userJson);
        if (user.username) return `mcq_validations_${user.username}`;
      } catch {}
    }
    return 'mcq_validations_anonymous';
  }

  /**
   * Cle composite pour identifier un MCQ unique
   */
  makeKey(mcqId: string, model: string): string {
    return `${mcqId}::${model}`;
  }

  /**
   * Sauvegarder une validation localement
   */
  saveValidation(mcqId: string, model: string, decision: 'ACCEPT' | 'REJECT'): void {
    const validations = this.getAllValidations();
    const key = this.makeKey(mcqId, model);
    validations[key] = {
      decision,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem(this.getStorageKey(), JSON.stringify(validations));
  }

  /**
   * Recuperer toutes les validations
   */
  getAllValidations(): Record<string, { decision: string, timestamp: string }> {
    const saved = localStorage.getItem(this.getStorageKey());
    return saved ? JSON.parse(saved) : {};
  }

  /**
   * Verifier si un MCQ a ete valide
   */
  isValidated(mcqId: string, model: string): boolean {
    const validations = this.getAllValidations();
    return this.makeKey(mcqId, model) in validations;
  }

  /**
   * Obtenir la decision pour un MCQ
   */
  getDecision(mcqId: string, model: string): 'ACCEPT' | 'REJECT' | null {
    const validations = this.getAllValidations();
    const key = this.makeKey(mcqId, model);
    return validations[key]?.decision as 'ACCEPT' | 'REJECT' || null;
  }

  /**
   * Effacer toutes les validations
   */
  clearAll(): void {
    localStorage.removeItem(this.getStorageKey());
  }

  /**
   * Recuperer les stats depuis le backend
   */
  getStatsFromBackend(): Observable<{
    total: number,
    pending: number,
    completed: number,
    accepted: number,
    rejected: number
  }> {
    return this.http.get<any>(`${this.apiUrl}/validations/stats`).pipe(
      catchError(() => {
        return of({ total: 0, pending: 0, completed: 0, accepted: 0, rejected: 0 });
      })
    );
  }

  /**
   * Synchroniser les validations avec le backend
   * Le backend renvoie des cles mcq_id::model
   */
  syncWithBackend(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/validations/user`).pipe(
      tap(response => {
        if (response.validations) {
          localStorage.setItem(this.getStorageKey(), JSON.stringify(response.validations));
        }
      }),
      catchError(error => {
        console.warn('Erreur sync backend:', error);
        return of(null);
      })
    );
  }

  /**
   * Sauvegarder une validation (localStorage + backend)
   */
  saveValidationToBackend(
    mcqId: string,
    model: string,
    decision: 'ACCEPT' | 'REJECT',
    validationData: any
  ): Observable<any> {
    // Sauvegarder localement d'abord
    this.saveValidation(mcqId, model, decision);

    // Envoyer au backend avec model en query param
    return this.http.post<any>(
      `${this.apiUrl}/validations/${mcqId}/validate?model=${encodeURIComponent(model)}`,
      validationData
    ).pipe(
      catchError(error => {
        console.warn('Backend indisponible, validation reste en local:', error);
        return of({ status: 'local', message: 'Saved locally only' });
      })
    );
  }
}
