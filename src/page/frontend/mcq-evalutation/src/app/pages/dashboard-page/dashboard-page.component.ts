import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { User } from '../../models';

/**
 * DashboardPage Component - Page principale après connexion
 */
@Component({
  selector: 'app-dashboard-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dashboard-page.component.html',
  styleUrl: './dashboard-page.component.scss'
})
export class DashboardPageComponent implements OnInit {
  private readonly authService = inject(AuthService);
  private readonly router = inject(Router);

  // État du composant
  currentUser = signal<User | null>(null);
  loading = signal(true);

  // Statistiques mock
  stats = {
    total: 0,
    pending: 0,
    completed: 0,
    accepted: 0,
    rejected: 0
  };

  ngOnInit(): void {
    // Charger l'utilisateur actuel
    this.authService.currentUser$.subscribe(user => {
      this.currentUser.set(user);
      this.loading.set(false);
    });

    // TODO: Charger les vraies statistiques depuis l'API
    this.loadMockStats();
  }

  /**
   * Charger des statistiques mock
   */
  private loadMockStats(): void {
    this.stats = {
      total: 50,
      pending: 30,
      completed: 20,
      accepted: 15,
      rejected: 5
    };
  }

  /**
   * Déconnexion
   */
  onLogout(): void {
    if (confirm('Voulez-vous vraiment vous déconnecter ?')) {
      this.authService.logout().subscribe({
        next: () => {
          console.log('Logged out successfully');
        },
        error: (error) => {
          console.error('Logout error:', error);
          // Même en cas d'erreur, on redirige vers login
        }
      });
    }
  }

  /**
   * Commencer l'évaluation
   */
  onStartEvaluation(): void {
    this.router.navigate(['/evaluation']);
  }
}
