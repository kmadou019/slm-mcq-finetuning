import { Component, inject, OnInit, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, NavigationEnd } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ValidationService } from '../../services/validation.service';
import { MCQAssignment, User } from '../../models';
import { McqSelectionModalComponent } from '../../components/mcq-selection-modal/mcq-selection-modal.component';
import { GenerateFromMaterialModalComponent } from '../../components/generate-from-material-modal/generate-from-material-modal.component';
import { filter, Subscription } from 'rxjs';

/**
 * DashboardPage Component - Page principale après connexion
 */
@Component({
  selector: 'app-dashboard-page',
  standalone: true,
  imports: [CommonModule, FormsModule, McqSelectionModalComponent, GenerateFromMaterialModalComponent],
  templateUrl: './dashboard-page.component.html',
  styleUrl: './dashboard-page.component.scss'
})
export class DashboardPageComponent implements OnInit, OnDestroy {
  private readonly authService = inject(AuthService);
  private readonly validationService = inject(ValidationService);
  private readonly router = inject(Router);
  private routerSubscription?: Subscription;

  // État du composant
  currentUser = signal<User | null>(null);
  loading = signal(true);
  showMcqModal = signal(false);
  showGenerateModal = signal(false);
  showPasswordModal = signal(false);
  passwordForm = {
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  };

  // Statistiques
  stats = signal({
    total: 0,
    pending: 0,
    completed: 0,
    accepted: 0,
    rejected: 0
  });

  // IDs des derniers QCMs assignés (pour évaluation ciblée)
  newMcqIds = signal<string[]>(this.loadNewMcqIds());

  ngOnInit(): void {
    // Charger l'utilisateur actuel
    this.authService.currentUser$.subscribe(user => {
      this.currentUser.set(user);
    });

    // Synchroniser avec le backend PUIS charger les stats (séquentiel)
    this.syncThenLoadStats();

    // Écouter les événements de navigation pour recharger les stats
    // quand l'utilisateur revient au dashboard depuis une autre page
    this.routerSubscription = this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: any) => {
      if (event.url === '/dashboard' || event.url === '/') {
        this.syncThenLoadStats();
      }
    });
  }

  /**
   * Synchroniser les validations avec le backend, puis charger les stats
   */
  private syncThenLoadStats(): void {
    this.loading.set(true);
    this.validationService.syncWithBackend().subscribe({
      next: () => this.loadStats(),
      error: () => this.loadStats()
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
        console.error('Erreur lors du chargement des stats:', error);
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
   * Commencer l'évaluation (tous les en-attente)
   */
  onStartEvaluation(): void {
    this.router.navigate(['/evaluation']);
  }

  /**
   * Commencer l'évaluation uniquement sur les nouveaux QCMs assignés
   */
  onStartEvaluationNewOnly(): void {
    this.router.navigate(['/evaluation'], { queryParams: { newOnly: 'true' } });
  }

  /**
   * Ouvrir la modal pour demander de nouveaux MCQ
   */
  onRequestNewMcqs(): void {
    this.showMcqModal.set(true);
  }

  /**
   * Ouvrir la modal de génération depuis contenu custom
   */
  onGenerateFromMaterial(): void {
    this.showGenerateModal.set(true);
  }

  /**
   * Fermer la modal de génération
   */
  onGenerateModalClose(): void {
    this.showGenerateModal.set(false);
  }

  /**
   * Génération terminée : recharger les stats
   */
  onGenerateModalConfirm(): void {
    this.showGenerateModal.set(false);
    this.router.navigate(['/evaluation']);
  }

  /**
   * Gérer la fermeture de la modal
   */
  onModalClose(): void {
    this.showMcqModal.set(false);
  }

  /**
   * Gérer la confirmation de la modal
   * Stocker les IDs assignés pour évaluation ciblée
   */
  onModalConfirm(assignment: MCQAssignment): void {
    console.log('Nouveaux MCQ assignés:', assignment.mcq_count);
    this.showMcqModal.set(false);
    if (assignment.assigned_mcq_ids?.length) {
      localStorage.setItem('lastAssignedMcqs', JSON.stringify(assignment.assigned_mcq_ids));
      this.newMcqIds.set(assignment.assigned_mcq_ids);
    }
    this.syncThenLoadStats();
  }

  private loadNewMcqIds(): string[] {
    try {
      return JSON.parse(localStorage.getItem('lastAssignedMcqs') ?? '[]') ?? [];
    } catch {
      return [];
    }
  }

  /**
   * Ouvrir la modal de changement de mot de passe
   */
  openPasswordModal(): void {
    this.passwordForm = { currentPassword: '', newPassword: '', confirmPassword: '' };
    this.showPasswordModal.set(true);
  }

  /**
   * Fermer la modal de changement de mot de passe
   */
  closePasswordModal(): void {
    this.showPasswordModal.set(false);
  }

  /**
   * Changer le mot de passe
   */
  changePassword(): void {
    if (!this.passwordForm.currentPassword || !this.passwordForm.newPassword) {
      alert('Veuillez remplir tous les champs');
      return;
    }

    if (this.passwordForm.newPassword !== this.passwordForm.confirmPassword) {
      alert('Les mots de passe ne correspondent pas');
      return;
    }

    this.authService.changePassword(
      this.passwordForm.currentPassword,
      this.passwordForm.newPassword
    ).subscribe({
      next: () => {
        alert('Mot de passe modifie avec succes !');
        this.closePasswordModal();
      },
      error: (error) => {
        alert(`Erreur: ${error.error?.detail || error.message}`);
      }
    });
  }

  /**
   * Aller au dashboard admin
   */
  onGoToAdmin(): void {
    this.router.navigate(['/admin']);
  }

  onGoToHistory(): void {
    this.router.navigate(['/history']);
  }

  onStatCardClick(filter?: string): void {
    if (filter === 'pending') {
      this.router.navigate(['/evaluation']);
    } else if (filter === 'accepted' || filter === 'rejected') {
      this.router.navigate(['/history'], { queryParams: { filter } });
    } else {
      this.router.navigate(['/history']);
    }
  }

  onGoToAnswerability(): void {
    this.router.navigate(['/answerability']);
  }
}
