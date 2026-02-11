import { inject } from '@angular/core';
import { Router, CanActivateFn } from '@angular/router';
import { AuthService } from '../services/auth.service';
import { map } from 'rxjs/operators';

/**
 * Guard pour protéger les routes admin
 * Seuls les utilisateurs avec role="admin" peuvent accéder
 */
export const adminGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  return authService.currentUser$.pipe(
    map(user => {
      if (user && user.role === 'admin') {
        return true;
      } else {
        console.warn('⛔ Accès admin refusé - redirection vers dashboard');
        router.navigate(['/dashboard']);
        return false;
      }
    })
  );
};
