import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { BehaviorSubject, Observable, tap, catchError, throwError } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  User,
  LoginRequest,
  LoginResponse,
  MCQAssignment,
  MCQSelectionRequest
} from '../models';

/**
 * Service for authentication and user management
 */
@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);
  private readonly apiUrl = environment.apiUrl;
  private readonly tokenKey = environment.tokenKey;

  // Observable pour l'utilisateur actuel
  private currentUserSubject = new BehaviorSubject<User | null>(
    this.getUserFromStorage()
  );
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor() {
    // Charger l'utilisateur depuis le storage au démarrage
    const user = this.getUserFromStorage();
    if (user) {
      this.currentUserSubject.next(user);
    }
  }

  /**
   * Connexion de l'utilisateur
   */
  login(credentials: LoginRequest): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.apiUrl}/auth/login`, credentials).pipe(
      tap(response => {
        // Sauvegarder le token et l'utilisateur
        this.saveToken(response.access_token);
        this.saveUser(response.user);
        this.currentUserSubject.next(response.user);
      }),
      catchError(error => {
        console.error('Login error:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Déconnexion de l'utilisateur
   */
  logout(): Observable<void> {
    console.log('🔄 AuthService.logout() appelé');

    return this.http.post<void>(`${this.apiUrl}/auth/logout`, {}).pipe(
      tap(() => {
        console.log('✅ Requête logout réussie');
        this.clearSession();
        console.log('✅ Session nettoyée, navigation vers /login');
        this.router.navigate(['/login']);
      }),
      catchError(error => {
        console.warn('⚠️ Erreur backend lors du logout, déconnexion locale quand même', error);
        // Même en cas d'erreur backend, on déconnecte localement
        this.clearSession();
        console.log('✅ Session nettoyée (après erreur), navigation vers /login');
        this.router.navigate(['/login']);
        // Retourner un Observable vide au lieu d'une erreur pour que le subscribe se termine proprement
        return new Observable<void>(observer => {
          observer.complete();
        });
      })
    );
  }

  /**
   * Récupérer l'utilisateur actuel depuis l'API
   */
  getCurrentUser(): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/auth/me`).pipe(
      tap(user => {
        this.saveUser(user);
        this.currentUserSubject.next(user);
      }),
      catchError(error => {
        console.error('Get current user error:', error);
        this.clearSession();
        return throwError(() => error);
      })
    );
  }

  /**
   * Assigner des MCQ à l'utilisateur actuel
   */
  assignMCQs(count: number, model?: string): Observable<MCQAssignment> {
    const request: MCQSelectionRequest = { count, model };
    return this.http.post<MCQAssignment>(`${this.apiUrl}/auth/assign-mcq`, request).pipe(
      catchError(error => {
        console.error('Assign MCQs error:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Vérifier si l'utilisateur est authentifié et le token n'est pas expiré
   */
  isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) {
      return false;
    }

    try {
      // Decode JWT and check expiration (simple base64 decode of payload)
      const payload = JSON.parse(atob(token.split('.')[1]));
      const exp = payload.exp;

      if (!exp) {
        return false;
      }

      // Check if token is expired (exp is in seconds, Date.now() is in ms)
      if (exp * 1000 < Date.now()) {
        console.warn('Token expired, clearing session');
        this.clearSession();
        return false;
      }

      return true;
    } catch (error) {
      console.error('Error validating token:', error);
      this.clearSession();
      return false;
    }
  }

  /**
   * Récupérer le token JWT
   */
  getToken(): string | null {
    return sessionStorage.getItem(this.tokenKey);
  }

  /**
   * Sauvegarder le token JWT
   */
  saveToken(token: string): void {
    sessionStorage.setItem(this.tokenKey, token);
  }

  /**
   * Supprimer le token JWT
   */
  removeToken(): void {
    sessionStorage.removeItem(this.tokenKey);
  }

  /**
   * Récupérer l'utilisateur actuel (valeur synchrone)
   */
  get currentUserValue(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * Sauvegarder l'utilisateur dans le sessionStorage
   */
  private saveUser(user: User): void {
    sessionStorage.setItem('mcq_current_user', JSON.stringify(user));
  }

  /**
   * Récupérer l'utilisateur depuis le sessionStorage
   */
  private getUserFromStorage(): User | null {
    const userJson = sessionStorage.getItem('mcq_current_user');
    if (userJson) {
      try {
        return JSON.parse(userJson);
      } catch (error) {
        console.error('Error parsing user from storage:', error);
        return null;
      }
    }
    return null;
  }

  /**
   * Supprimer l'utilisateur du sessionStorage
   */
  private removeUser(): void {
    sessionStorage.removeItem('mcq_current_user');
  }

  /**
   * Nettoyer toute la session (token + user)
   */
  private clearSession(): void {
    // Nettoyer les données d'évaluation spécifiques à l'utilisateur
    const user = this.getUserFromStorage();
    if (user?.username) {
      localStorage.removeItem(`mcq_validations_${user.username}`);
      localStorage.removeItem(`mcq_evaluation_progress_${user.username}`);
    }
    this.removeToken();
    this.removeUser();
    this.currentUserSubject.next(null);
  }
}
