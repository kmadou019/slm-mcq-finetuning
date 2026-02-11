import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { LoginRequest } from '../../models';

/**
 * LoginPage Component - Page de connexion des évaluateurs
 */
@Component({
  selector: 'app-login-page',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './login-page.component.html',
  styleUrl: './login-page.component.scss'
})
export class LoginPageComponent {
  private readonly fb = inject(FormBuilder);
  private readonly authService = inject(AuthService);
  private readonly router = inject(Router);

  // Formulaire de connexion
  loginForm: FormGroup;

  // État du composant
  loading = signal(false);
  error = signal<string | null>(null);

  constructor() {
    // Initialiser le formulaire
    this.loginForm = this.fb.group({
      username: ['', [Validators.required, Validators.minLength(3)]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });

    // Rediriger si déjà connecté
    if (this.authService.isAuthenticated()) {
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * Soumettre le formulaire de connexion
   */
  onSubmit(): void {
    // Réinitialiser l'erreur
    this.error.set(null);

    // Vérifier la validité du formulaire
    if (this.loginForm.invalid) {
      this.markFormGroupTouched(this.loginForm);
      return;
    }

    // Récupérer les valeurs du formulaire
    const credentials: LoginRequest = this.loginForm.value;

    // Démarrer le chargement
    this.loading.set(true);

    // Appeler le service d'authentification
    this.authService.login(credentials).subscribe({
      next: (response) => {
        console.log('Login successful:', response);
        // Rediriger directement vers le dashboard
        this.loading.set(false);
        this.router.navigate(['/dashboard']);
      },
      error: (error) => {
        console.error('Login error:', error);
        this.loading.set(false);

        // Gérer les différents types d'erreurs
        if (error.status === 401) {
          this.error.set('Identifiants incorrects. Veuillez réessayer.');
        } else if (error.status === 0) {
          this.error.set('Impossible de se connecter au serveur. Vérifiez votre connexion.');
        } else {
          this.error.set('Une erreur est survenue. Veuillez réessayer plus tard.');
        }
      }
    });
  }


  /**
   * Marquer tous les champs du formulaire comme "touched"
   * pour afficher les erreurs de validation
   */
  private markFormGroupTouched(formGroup: FormGroup): void {
    Object.keys(formGroup.controls).forEach(key => {
      const control = formGroup.get(key);
      control?.markAsTouched();
    });
  }

  /**
   * Vérifier si un champ a une erreur et est touché
   */
  hasError(fieldName: string): boolean {
    const field = this.loginForm.get(fieldName);
    return !!(field && field.invalid && field.touched);
  }

  /**
   * Récupérer le message d'erreur pour un champ
   */
  getErrorMessage(fieldName: string): string {
    const field = this.loginForm.get(fieldName);
    if (!field) return '';

    if (field.hasError('required')) {
      return `Le champ ${fieldName === 'username' ? 'nom d\'utilisateur' : 'mot de passe'} est requis.`;
    }

    if (field.hasError('minlength')) {
      const minLength = field.errors?.['minlength'].requiredLength;
      return `Minimum ${minLength} caractères requis.`;
    }

    return '';
  }
}
