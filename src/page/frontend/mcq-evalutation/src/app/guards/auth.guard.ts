import { inject } from '@angular/core';
import { Router, CanActivateFn } from '@angular/router';
import { AuthService } from '../services/auth.service';

/**
 * AuthGuard - Protection des routes
 * Vérifie si l'utilisateur est authentifié avant d'accéder à une route
 */
export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  // Vérifier si l'utilisateur est authentifié
  if (authService.isAuthenticated()) {
    return true;
  }

  // Rediriger vers la page de login avec l'URL de retour
  router.navigate(['/login'], {
    queryParams: { returnUrl: state.url }
  });

  return false;
};
