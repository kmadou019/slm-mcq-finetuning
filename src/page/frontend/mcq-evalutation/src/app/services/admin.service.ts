import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * Service pour gérer les fonctionnalités admin
 */
@Injectable({
  providedIn: 'root'
})
export class AdminService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api/admin';

  // ============================================================================
  // STATISTIQUES
  // ============================================================================

  /**
   * Récupérer les statistiques globales
   */
  getGlobalStats(): Observable<{
    total_validations: number;
    accepted: number;
    rejected: number;
    acceptance_rate: number;
    active_evaluators: number;
    total_assignments: number;
    completion_rate: number;
  }> {
    return this.http.get<any>(`${this.apiUrl}/stats/global`);
  }

  /**
   * Récupérer les statistiques par modèle
   */
  getStatsByModel(): Observable<Array<{
    model: string;
    total_available: number;
    total_assigned: number;
    evaluated: number;
    accepted: number;
    rejected: number;
    remaining: number;
    acceptance_rate: number;
    progress_percent: number;
  }>> {
    return this.http.get<any>(`${this.apiUrl}/stats/by-model`);
  }

  // ============================================================================
  // GESTION UTILISATEURS
  // ============================================================================

  /**
   * Lister tous les utilisateurs avec leurs stats
   */
  listUsers(): Observable<Array<{
    id: string;
    username: string;
    email: string;
    role: string;
    created_at: string;
    stats: {
      assignments: number;
      validations: number;
      accepted: number;
      rejected: number;
    };
  }>> {
    return this.http.get<any>(`${this.apiUrl}/users`);
  }

  /**
   * Créer un nouvel utilisateur
   */
  createUser(userData: {
    username: string;
    password: string;
    email: string;
    role: string;
  }): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/users`, userData);
  }

  /**
   * Modifier un utilisateur
   */
  updateUser(userId: string, data: { email?: string; password?: string; role?: string }): Observable<any> {
    return this.http.put<any>(`${this.apiUrl}/users/${userId}`, data);
  }

  /**
   * Supprimer un utilisateur
   */
  deleteUser(userId: string): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/users/${userId}`);
  }

  // ============================================================================
  // GESTION TRACKER
  // ============================================================================

  /**
   * Récupérer le tracker global
   */
  getTracker(): Observable<{
    tracker: any;
    path: string;
    exists: boolean;
  }> {
    return this.http.get<any>(`${this.apiUrl}/tracker`);
  }

  /**
   * Mettre à jour le tracker global
   */
  updateTracker(trackerData: any): Observable<any> {
    return this.http.put<any>(`${this.apiUrl}/tracker`, trackerData);
  }

  /**
   * Réinitialiser le tracker pour un modèle
   */
  resetTrackerForModel(model: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/tracker/reset/${model}`, {});
  }

  // ============================================================================
  // RESET
  // ============================================================================

  resetAllData(): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/reset-all`);
  }

  // ============================================================================
  // EXPORT
  // ============================================================================

  exportCsv(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/export/csv`, { responseType: 'blob' });
  }

  exportSft(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/export/sft`, { responseType: 'blob' });
  }

  exportDpo(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/export/dpo`, { responseType: 'blob' });
  }

  exportKto(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/export/kto`, { responseType: 'blob' });
  }

  exportStats(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/export/stats`);
  }
}
