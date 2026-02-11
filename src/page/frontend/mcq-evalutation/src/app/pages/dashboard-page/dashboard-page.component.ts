import { Component, inject, OnInit, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { McqService } from '../../services/mcq.service';
import { ValidationService } from '../../services/validation.service';
import { User } from '../../models';
import { McqSelectionModalComponent } from '../../components/mcq-selection-modal/mcq-selection-modal.component';
import { filter, Subscription } from 'rxjs';

/**
 * DashboardPage Component - Page principale après connexion
 */
@Component({
  selector: 'app-dashboard-page',
  standalone: true,
  imports: [CommonModule, McqSelectionModalComponent],
  templateUrl: './dashboard-page.component.html',
  styleUrl: './dashboard-page.component.scss'
})
export class DashboardPageComponent implements OnInit, OnDestroy {
  private readonly authService = inject(AuthService);
  private readonly mcqService = inject(McqService);
  private readonly validationService = inject(ValidationService);
  private readonly router = inject(Router);
  private routerSubscription?: Subscription;

  // État du composant
  currentUser = signal<User | null>(null);
  loading = signal(true);
  showMcqModal = signal(false);

  // Statistiques
  stats = signal({
    total: 0,
    pending: 0,
    completed: 0,
    accepted: 0,
    rejected: 0
  });

  ngOnInit(): void {
    // Charger l'utilisateur actuel
    this.authService.currentUser$.subscribe(user => {
      this.currentUser.set(user);
    });

    // Synchroniser avec le backend (récupérer les validations)
    this.validationService.syncWithBackend().subscribe();

    // Charger les vraies statistiques depuis l'API
    this.loadStats();

    // Écouter les événements de navigation pour recharger les stats
    // quand l'utilisateur revient au dashboard depuis une autre page
    this.routerSubscription = this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: any) => {
      if (event.url === '/dashboard' || event.url === '/') {
        console.log('🔄 Retour au dashboard - rechargement des stats');
        // Re-synchroniser et recharger les stats
        this.validationService.syncWithBackend().subscribe(() => {
          this.loadStats();
        });
      }
    });
  }

  ngOnDestroy(): void {
    // Nettoyer la souscription pour éviter les fuites mémoire
    this.routerSubscription?.unsubscribe();
  }

  /**
   * Charger les statistiques réelles depuis le backend
   */
  private loadStats(): void {
    this.loading.set(true);

    // Charger les stats depuis le backend
    this.validationService.getStatsFromBackend().subscribe({
      next: (stats) => {
        console.log('📊 Stats chargées depuis backend:', stats);
        this.stats.set(stats);
        this.loading.set(false);
      },
      error: (error) => {
        console.error('❌ Erreur lors du chargement des stats:', error);
        // En cas d'erreur, fallback sur localStorage
        this.mcqService.getAssignedMcqs().subscribe({
          next: (response) => {
            const localStats = this.validationService.getStats(response.mcq_ids);
            console.log('📊 Stats depuis localStorage (fallback):', localStats);
            this.stats.set(localStats);
            this.loading.set(false);
          },
          error: () => {
            // Si tout échoue, stats vides
            this.stats.set({
              total: 0,
              pending: 0,
              completed: 0,
              accepted: 0,
              rejected: 0
            });
            this.loading.set(false);
          }
        });
      }
    });
  }

  /**
   * Déconnexion
   */
  onLogout(): void {
    if (confirm('Voulez-vous vraiment vous déconnecter ?')) {
      console.log('🔄 Déconnexion en cours...');
      this.authService.logout().subscribe({
        next: () => {
          console.log('✅ Déconnexion réussie - Redirection vers /login');
          // La redirection est déjà gérée par le service AuthService
        },
        error: (error) => {
          console.error('❌ Erreur lors de la déconnexion:', error);
          // La redirection est déjà gérée par le service AuthService même en cas d'erreur
        }
      });
    } else {
      console.log('❌ Déconnexion annulée par l\'utilisateur');
    }
  }

  /**
   * Commencer l'évaluation
   */
  onStartEvaluation(): void {
    this.router.navigate(['/evaluation']);
  }

  /**
   * Ouvrir la modal pour demander de nouveaux MCQ
   */
  onRequestNewMcqs(): void {
    this.showMcqModal.set(true);
  }

  /**
   * Gérer la fermeture de la modal
   */
  onModalClose(): void {
    this.showMcqModal.set(false);
  }

  /**
   * Gérer la confirmation de la modal
   * Recharger les stats après attribution de nouveaux MCQ
   */
  onModalConfirm(count: number): void {
    console.log('Nouveaux MCQ assignés:', count);
    this.showMcqModal.set(false);
    // Recharger les stats pour afficher les nouveaux MCQ
    this.loadStats();
  }

  /**
   * Aller au dashboard admin
   */
  onGoToAdmin(): void {
    this.router.navigate(['/admin']);
  }
}
