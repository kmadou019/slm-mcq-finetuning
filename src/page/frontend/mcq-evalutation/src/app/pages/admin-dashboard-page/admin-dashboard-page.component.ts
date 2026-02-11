import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AdminService } from '../../services/admin.service';
import { AuthService } from '../../services/auth.service';

/**
 * AdminDashboardPage Component - Dashboard administrateur
 */
@Component({
  selector: 'app-admin-dashboard-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './admin-dashboard-page.component.html',
  styleUrl: './admin-dashboard-page.component.scss'
})
export class AdminDashboardPageComponent implements OnInit {
  private readonly adminService = inject(AdminService);
  private readonly authService = inject(AuthService);
  private readonly router = inject(Router);

  // État du composant
  loading = signal(true);
  activeTab = signal<'stats' | 'users' | 'tracker' | 'export'>('stats');

  // Statistiques globales
  globalStats = signal<any>(null);

  // Statistiques par modèle
  modelStats = signal<any[]>([]);

  // Utilisateurs
  users = signal<any[]>([]);
  showCreateUserModal = signal(false);
  newUser = signal({
    username: '',
    password: '',
    email: '',
    role: 'evaluator'
  });

  // Tracker
  tracker = signal<any>(null);
  trackerPath = signal('');
  showTrackerEditor = signal(false);
  trackerJson = signal('');

  ngOnInit(): void {
    this.loadAllData();
  }

  /**
   * Charger toutes les données
   */
  private loadAllData(): void {
    this.loading.set(true);

    // Charger les stats globales
    this.adminService.getGlobalStats().subscribe({
      next: (stats) => {
        console.log('📊 Global stats:', stats);
        this.globalStats.set(stats);
      },
      error: (error) => {
        console.error('❌ Erreur chargement stats globales:', error);
      }
    });

    // Charger les stats par modèle
    this.adminService.getStatsByModel().subscribe({
      next: (stats) => {
        console.log('📊 Model stats:', stats);
        this.modelStats.set(stats);
      },
      error: (error) => {
        console.error('❌ Erreur chargement stats par modèle:', error);
      }
    });

    // Charger les utilisateurs
    this.adminService.listUsers().subscribe({
      next: (users) => {
        console.log('👥 Users:', users);
        this.users.set(users);
      },
      error: (error) => {
        console.error('❌ Erreur chargement utilisateurs:', error);
      }
    });

    // Charger le tracker
    this.adminService.getTracker().subscribe({
      next: (data) => {
        console.log('🎯 Tracker:', data);
        this.tracker.set(data.tracker);
        this.trackerPath.set(data.path);
        this.trackerJson.set(JSON.stringify(data.tracker, null, 2));
        this.loading.set(false);
      },
      error: (error) => {
        console.error('❌ Erreur chargement tracker:', error);
        this.loading.set(false);
      }
    });
  }

  /**
   * Changer d'onglet
   */
  setActiveTab(tab: 'stats' | 'users' | 'tracker' | 'export'): void {
    this.activeTab.set(tab);
  }

  // ============================================================================
  // GESTION UTILISATEURS
  // ============================================================================

  /**
   * Ouvrir la modal de création d'utilisateur
   */
  openCreateUserModal(): void {
    this.newUser.set({
      username: '',
      password: '',
      email: '',
      role: 'evaluator'
    });
    this.showCreateUserModal.set(true);
  }

  /**
   * Fermer la modal de création d'utilisateur
   */
  closeCreateUserModal(): void {
    this.showCreateUserModal.set(false);
  }

  /**
   * Créer un nouvel utilisateur
   */
  createUser(): void {
    const userData = this.newUser();

    if (!userData.username || !userData.password || !userData.email) {
      alert('Veuillez remplir tous les champs requis');
      return;
    }

    this.adminService.createUser(userData).subscribe({
      next: (response) => {
        console.log('✅ Utilisateur créé:', response);
        alert(`Utilisateur ${userData.username} créé avec succès !`);
        this.closeCreateUserModal();
        // Recharger la liste des utilisateurs
        this.adminService.listUsers().subscribe(users => {
          this.users.set(users);
        });
      },
      error: (error) => {
        console.error('❌ Erreur création utilisateur:', error);
        alert(`Erreur lors de la création: ${error.error?.detail || error.message}`);
      }
    });
  }

  /**
   * Supprimer un utilisateur
   */
  deleteUser(userId: string, username: string): void {
    if (!confirm(`Voulez-vous vraiment supprimer l'utilisateur ${username} ?`)) {
      return;
    }

    this.adminService.deleteUser(userId).subscribe({
      next: (response) => {
        console.log('✅ Utilisateur supprimé:', response);
        alert(`Utilisateur ${username} supprimé avec succès !`);
        // Recharger la liste des utilisateurs
        this.adminService.listUsers().subscribe(users => {
          this.users.set(users);
        });
      },
      error: (error) => {
        console.error('❌ Erreur suppression utilisateur:', error);
        alert(`Erreur lors de la suppression: ${error.error?.detail || error.message}`);
      }
    });
  }

  // ============================================================================
  // GESTION TRACKER
  // ============================================================================

  /**
   * Ouvrir l'éditeur de tracker
   */
  openTrackerEditor(): void {
    this.showTrackerEditor.set(true);
  }

  /**
   * Fermer l'éditeur de tracker
   */
  closeTrackerEditor(): void {
    this.showTrackerEditor.set(false);
  }

  /**
   * Sauvegarder le tracker
   */
  saveTracker(): void {
    try {
      const trackerData = JSON.parse(this.trackerJson());

      this.adminService.updateTracker(trackerData).subscribe({
        next: (response) => {
          console.log('✅ Tracker mis à jour:', response);
          alert('Tracker mis à jour avec succès !');
          this.tracker.set(trackerData);
          this.closeTrackerEditor();
        },
        error: (error) => {
          console.error('❌ Erreur mise à jour tracker:', error);
          alert(`Erreur lors de la mise à jour: ${error.error?.detail || error.message}`);
        }
      });
    } catch (error) {
      alert('JSON invalide ! Veuillez corriger la syntaxe.');
    }
  }

  /**
   * Réinitialiser le tracker pour un modèle
   */
  resetModelTracker(model: string): void {
    if (!confirm(`Voulez-vous vraiment réinitialiser le tracker pour le modèle ${model} ?`)) {
      return;
    }

    this.adminService.resetTrackerForModel(model).subscribe({
      next: (response) => {
        console.log('✅ Tracker réinitialisé:', response);
        alert(`Tracker réinitialisé pour ${model} !`);
        // Recharger le tracker
        this.adminService.getTracker().subscribe(data => {
          this.tracker.set(data.tracker);
          this.trackerJson.set(JSON.stringify(data.tracker, null, 2));
        });
        // Recharger les stats
        this.adminService.getStatsByModel().subscribe(stats => {
          this.modelStats.set(stats);
        });
      },
      error: (error) => {
        console.error('❌ Erreur réinitialisation tracker:', error);
        alert(`Erreur: ${error.error?.detail || error.message}`);
      }
    });
  }

  // ============================================================================
  // EXPORT
  // ============================================================================

  /**
   * Exporter les validations
   */
  exportValidations(): void {
    this.adminService.exportValidations().subscribe({
      next: (data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `validations_${new Date().toISOString()}.json`;
        link.click();
        window.URL.revokeObjectURL(url);
        console.log('✅ Validations exportées');
      },
      error: (error) => {
        console.error('❌ Erreur export validations:', error);
        alert('Erreur lors de l\'export');
      }
    });
  }

  /**
   * Exporter le rapport complet
   */
  exportFullReport(): void {
    this.adminService.exportStats().subscribe({
      next: (data) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `rapport_complet_${new Date().toISOString()}.json`;
        link.click();
        window.URL.revokeObjectURL(url);
        console.log('✅ Rapport complet exporté');
      },
      error: (error) => {
        console.error('❌ Erreur export rapport:', error);
        alert('Erreur lors de l\'export');
      }
    });
  }

  /**
   * Retourner au dashboard
   */
  goToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  /**
   * Déconnexion
   */
  onLogout(): void {
    if (confirm('Voulez-vous vraiment vous déconnecter ?')) {
      this.authService.logout().subscribe();
    }
  }
}
