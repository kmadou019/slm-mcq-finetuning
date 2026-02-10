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
    return this.http.post<void>(`${this.apiUrl}/auth/logout`, {}).pipe(
      tap(() => {
        this.clearSession();
        this.router.navigate(['/login']);
      }),
      catchError(error => {
        // Même en cas d'erreur, on déconnecte localement
        this.clearSession();
        this.router.navigate(['/login']);
        return throwError(() => error);
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
  assignMCQs(count: number): Observable<MCQAssignment> {
    const request: MCQSelectionRequest = { count };
    return this.http.post<MCQAssignment>(`${this.apiUrl}/auth/assign-mcq`, request).pipe(
      catchError(error => {
        console.error('Assign MCQs error:', error);
        return throwError(() => error);
      })
    );
  }

  /**
   * Vérifier si l'utilisateur est authentifié
   */
  isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) {
      return false;
    }

    // Optionnel: Vérifier si le token est expiré
    // Pour l'instant, on vérifie juste s'il existe
    return true;
  }

  /**
   * Récupérer le token JWT
   */
  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  /**
   * Sauvegarder le token JWT
   */
  saveToken(token: string): void {
    localStorage.setItem(this.tokenKey, token);
  }

  /**
   * Supprimer le token JWT
   */
  removeToken(): void {
    localStorage.removeItem(this.tokenKey);
  }

  /**
   * Récupérer l'utilisateur actuel (valeur synchrone)
   */
  get currentUserValue(): User | null {
    return this.currentUserSubject.value;
  }

  /**
   * Sauvegarder l'utilisateur dans le localStorage
   */
  private saveUser(user: User): void {
    localStorage.setItem('mcq_current_user', JSON.stringify(user));
  }

  /**
   * Récupérer l'utilisateur depuis le localStorage
   */
  private getUserFromStorage(): User | null {
    const userJson = localStorage.getItem('mcq_current_user');
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
   * Supprimer l'utilisateur du localStorage
   */
  private removeUser(): void {
    localStorage.removeItem('mcq_current_user');
  }

  /**
   * Nettoyer toute la session (token + user)
   */
  private clearSession(): void {
    this.removeToken();
    this.removeUser();
    this.currentUserSubject.next(null);
  }
}
