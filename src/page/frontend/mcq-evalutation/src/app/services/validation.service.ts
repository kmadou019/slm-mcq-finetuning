import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { tap, catchError } from 'rxjs/operators';

/**
 * Service pour gérer les validations
 * Synchronise avec le backend et utilise localStorage comme cache
 */
@Injectable({
  providedIn: 'root'
})
export class ValidationService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api';
  private STORAGE_KEY = 'mcq_validations';

  /**
   * Sauvegarder une validation
   */
  saveValidation(mcqId: string, decision: 'ACCEPT' | 'REJECT'): void {
    const validations = this.getAllValidations();
    validations[mcqId] = {
      decision,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(validations));
  }

  /**
   * Récupérer toutes les validations
   */
  getAllValidations(): Record<string, { decision: string, timestamp: string }> {
    const saved = localStorage.getItem(this.STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  }

  /**
   * Vérifier si un MCQ a été validé
   */
  isValidated(mcqId: string): boolean {
    const validations = this.getAllValidations();
    return mcqId in validations;
  }

  /**
   * Obtenir la décision pour un MCQ
   */
  getDecision(mcqId: string): 'ACCEPT' | 'REJECT' | null {
    const validations = this.getAllValidations();
    return validations[mcqId]?.decision as 'ACCEPT' | 'REJECT' || null;
  }

  /**
   * Obtenir les statistiques de validation
   */
  getStats(assignedMcqIds: string[]): {
    total: number,
    pending: number,
    completed: number,
    accepted: number,
    rejected: number
  } {
    const validations = this.getAllValidations();

    let completed = 0;
    let accepted = 0;
    let rejected = 0;

    assignedMcqIds.forEach(mcqId => {
      if (validations[mcqId]) {
        completed++;
        if (validations[mcqId].decision === 'ACCEPT') {
          accepted++;
        } else {
          rejected++;
        }
      }
    });

    return {
      total: assignedMcqIds.length,
      pending: assignedMcqIds.length - completed,
      completed,
      accepted,
      rejected
    };
  }

  /**
   * Effacer toutes les validations
   */
  clearAll(): void {
    localStorage.removeItem(this.STORAGE_KEY);
  }

  /**
   * Récupérer les stats depuis le backend
   * Fallback sur localStorage si le backend n'est pas disponible
   */
  getStatsFromBackend(): Observable<{
    total: number,
    pending: number,
    completed: number,
    accepted: number,
    rejected: number
  }> {
    return this.http.get<any>(`${this.apiUrl}/validations/stats`).pipe(
      tap(stats => {
        console.log('📊 Stats depuis backend:', stats);
      }),
      catchError(error => {
        console.warn('⚠️ Backend indisponible, utilisation de localStorage', error);
        // Fallback: retourner des stats vides
        return of({
          total: 0,
          pending: 0,
          completed: 0,
          accepted: 0,
          rejected: 0
        });
      })
    );
  }

  /**
   * Synchroniser les validations avec le backend
   * Récupérer toutes les validations de l'utilisateur et mettre à jour localStorage
   */
  syncWithBackend(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/validations/user`).pipe(
      tap(response => {
        console.log('🔄 Sync backend → localStorage:', response);
        // Mettre à jour localStorage avec les données du backend
        if (response.validations) {
          localStorage.setItem(this.STORAGE_KEY, JSON.stringify(response.validations));
        }
      }),
      catchError(error => {
        console.warn('⚠️ Erreur sync backend:', error);
        return of(null);
      })
    );
  }

  /**
   * Sauvegarder une validation (localStorage + backend)
   */
  saveValidationToBackend(
    mcqId: string,
    decision: 'ACCEPT' | 'REJECT',
    validationData: any
  ): Observable<any> {
    // Sauvegarder d'abord localement (rapide)
    this.saveValidation(mcqId, decision);

    // Puis envoyer au backend
    return this.http.post<any>(
      `${this.apiUrl}/validations/${mcqId}/validate`,
      validationData
    ).pipe(
      tap(response => {
        console.log('✅ Validation sauvegardée sur backend:', response);
      }),
      catchError(error => {
        console.warn('⚠️ Backend indisponible, validation reste en local:', error);
        // Retourner une réponse factice pour ne pas bloquer l'UI
        return of({ status: 'local', message: 'Saved locally only' });
      })
    );
  }
}
