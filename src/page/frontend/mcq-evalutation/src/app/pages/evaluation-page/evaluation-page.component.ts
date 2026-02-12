import { Component, inject, OnInit, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { Router, NavigationEnd } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { McqService } from '../../services/mcq.service';
import { ValidationService } from '../../services/validation.service';
import { MCQCard, SectionCheck } from '../../models';
import { filter, Subscription } from 'rxjs';

/**
 * EvaluationPage Component - Page d'évaluation des MCQ
 */
@Component({
  selector: 'app-evaluation-page',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './evaluation-page.component.html',
  styleUrl: './evaluation-page.component.scss'
})
export class EvaluationPageComponent implements OnInit, OnDestroy {
  private readonly fb = inject(FormBuilder);
  private readonly authService = inject(AuthService);
  private readonly mcqService = inject(McqService);
  private readonly validationService = inject(ValidationService);
  private readonly router = inject(Router);
  private routerSubscription?: Subscription;

  // État du composant
  loading = signal(true);
  currentIndex = signal(0);
  currentMcq = signal<MCQCard | null>(null);
  mcqList = signal<string[]>([]); // Liste des IDs de MCQ assignés

  // Formulaire de validation
  validationForm: FormGroup;

  constructor() {
    // Initialiser le formulaire de validation
    this.validationForm = this.fb.group({
      feedback: ['']
    });
  }

  ngOnInit(): void {
    // Initialiser le chargement
    this.initializeEvaluation();

    // Écouter les événements de navigation pour recharger les MCQ
    // quand l'utilisateur revient à la page evaluation après avoir demandé de nouveaux MCQ
    this.routerSubscription = this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: any) => {
      if (event.url === '/evaluation') {
        console.log('🔄 Retour à l\'évaluation - rechargement des MCQ');
        this.initializeEvaluation();
      }
    });
  }

  ngOnDestroy(): void {
    // Nettoyer la souscription pour éviter les fuites mémoire
    this.routerSubscription?.unsubscribe();
  }

  /**
   * Initialiser ou réinitialiser l'évaluation
   * Sync avec le backend d'abord pour avoir l'état à jour des validations
   */
  private initializeEvaluation(): void {
    this.loading.set(true);

    // D'abord synchroniser les validations avec le backend
    this.validationService.syncWithBackend().subscribe({
      next: () => this.afterSync(),
      error: () => this.afterSync()
    });
  }

  /**
   * Après la synchronisation, vérifier la progression ou charger les MCQs
   */
  private afterSync(): void {
    // Toujours charger la liste depuis le backend (source de verite)
    this.loadAssignedMCQs();
  }

  /**
   * Charger les MCQ assignés depuis le backend, en filtrant ceux déjà évalués
   * Utilise la progression sauvegardée pour reprendre à la bonne position
   */
  private loadAssignedMCQs(): void {
    this.loading.set(true);

    this.mcqService.getAssignedMcqs().subscribe({
      next: (response) => {
        // Filtrer les MCQs déjà évalués
        const validations = this.validationService.getAllValidations();
        const pendingMcqs = response.mcq_ids.filter(id => !validations[id]);

        console.log(`MCQs assignes: ${response.mcq_ids.length}, deja evalues: ${response.mcq_ids.length - pendingMcqs.length}, en attente: ${pendingMcqs.length}`);

        this.mcqList.set(pendingMcqs);

        if (pendingMcqs.length > 0) {
          // Verifier s'il y a une progression sauvegardee pour reprendre
          const savedProgress = this.mcqService.loadProgress();
          let startIndex = 0;

          if (savedProgress && savedProgress.mcq_list.length > 0) {
            // Trouver la position de reprise dans la nouvelle liste
            const lastMcqId = savedProgress.mcq_list[savedProgress.current_index];
            const resumeIndex = pendingMcqs.indexOf(lastMcqId);
            if (resumeIndex >= 0) {
              startIndex = resumeIndex;
              console.log(`Reprise a la position ${startIndex} (${lastMcqId})`);
            }
            this.mcqService.clearProgress();
          }

          this.loadMCQ(startIndex);
        } else {
          this.loading.set(false);
          this.mcqService.clearProgress();
          alert('Toutes les evaluations sont terminees ! Vous pouvez demander de nouveaux MCQ depuis le dashboard.');
          this.router.navigate(['/dashboard']);
        }
      },
      error: (error) => {
        console.error('Erreur backend, utilisation des donnees mockees:', error);
        this.loadMockData();
      }
    });
  }

  /**
   * Charger des données mockées (fallback si backend indisponible)
   */
  private loadMockData(): void {
    console.log('🔄 Chargement des données mockées (mode développement)');
    this.mcqList.set(['MCQ-000001', 'MCQ-000002', 'MCQ-000003']);
    this.loadMCQ_MOCK(0);
  }

  /**
   * Charger un MCQ spécifique depuis le backend
   */
  private loadMCQ(index: number): void {
    console.log('📄 loadMCQ - Loading MCQ at index:', index);

    this.loading.set(true);
    const mcqId = this.mcqList()[index];
    console.log('🔑 MCQ ID:', mcqId);

    this.mcqService.getMcqById(mcqId).subscribe({
      next: (mcq) => {
        console.log('✅ MCQ chargé:', mcq);
        this.currentMcq.set(mcq);
        this.currentIndex.set(index);
        this.loading.set(false);
        this.validationForm.reset();
      },
      error: (error) => {
        console.error('❌ Erreur backend, utilisation des données mockées:', error);
        // Fallback vers les données mockées
        this.loadMCQ_MOCK(index);
      }
    });
  }

  /**
   * Charger un MCQ spécifique (ANCIENNE VERSION MOCKÉE - GARDÉE POUR RÉFÉRENCE)
   */
  private loadMCQ_MOCK(index: number): void {
    console.log('📄 loadMCQ - Loading MCQ at index:', index);

    const mcqId = this.mcqList()[index];
    console.log('🔑 MCQ ID:', mcqId);

    // Mock data correspondant à mcq_cards.html - chargement immédiat
    const mockMCQ: MCQCard = {
      item_id: mcqId,
      source_material: 'OIC-005-01-A',
      generator_info: 'qwen3_8b_pdapt_slerp',
      output_format: 'JSON',
      mcq_question: 'Quel est l\'élément le plus difficile à prouver pour établir la responsabilité d\'un professionnel de santé?',
      options: {
        A: 'Le dommage',
        B: 'Le fait générateur de responsabilité',
        C: 'Le lien de causalité entre le fait générateur et le dommage',
        D: 'L\'indemnisation'
      },
      correct_option: 'C',
      section_a_checks: [
        {
          check_id: 'A1',
          description: 'Question mark present',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          threshold: 'required',
          notes: 'Not a question'
        },
        {
          check_id: 'A2',
          description: 'No leading negation',
          result: 'PASS',
          status: 'not_checked',
          confidence: null,
          threshold: 'required',
          notes: 'No negation'
        },
        {
          check_id: 'A3',
          description: 'Originality (integral)',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          threshold: '≥ 0.75',
          notes: 'Score: N/A'
        },
        {
          check_id: 'A4',
          description: 'Readability (FK grade)',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          threshold: '≥ 12',
          notes: 'Score: None'
        }
      ],
      section_b_checks: [
        {
          check_id: 'B1',
          description: 'Disclosure (answer leakage)',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          score: 'True/False',
          notes: 'Score: N/A'
        },
        {
          check_id: 'B2',
          description: 'Relevance to material',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          score: '0-1',
          notes: 'Score: N/A'
        },
        {
          check_id: 'B3',
          description: 'Distractor plausibility',
          result: 'WARN',
          status: 'not_checked',
          confidence: null,
          score: 'True/False',
          notes: 'Score: False'
        },
        {
          check_id: 'B4',
          description: 'Difficulty appropriateness',
          result: 'PASS',
          status: 'not_checked',
          confidence: null,
          score: 'low/med/high',
          notes: 'Judge: medium'
        }
      ],
      decision_policy: 'Accept if all hard constraints pass and no critical AI-judge dimension fails.',
      final_decision: 'REVISE',
      audit_trail: 'Judge model: Automated, consistency: evaluated from CSV metrics.',
      lisa_texte_brut: 'La responsabilité d\'un professionnel de santé ou d\'un établissement de santé peut être recherchée à deux fins : soit la sanction du professionnel ou de l\'établissement, soit l\'indemnisation de l\'usager victime des conséquences d\'un événement indésirable associé aux soins. La sanction peut être de nature pénale ou disciplinaire. L\'indemnisation incombe au responsable (exercice libéral) ou à son employeur s\'il est salarié (public ou privé). Elle peut également être obtenue via la procédure amiable devant les commissions de conciliation et d\'indemnisation. L\'indemnisation des victimes est assurée par l\'obligation de souscription pour les médecins et les établissements de santé (public ou privé) d\'une assurance de responsabilité civile professionnelle. En plus d\'assurer l\'indemnisation des victimes, cette assurance permet de couvrir les médecins salariés lorsque la faute est détachable du service ou lorsque la mission qui leur a été confiée est outrepassée.\n\nL\'indemnisation de l\'usager s\'estimant victime des conséquences dommageables d\'un acte médical suppose qu\'il apporte la preuve d\'un dommage, d\'un fait générateur de responsabilité et d\'un lien de causalité entre le fait générateur et le dommage.\n\nEn matière de responsabilité des professionnels et des établissements de santé, le dommage peut prendre plusieurs formes. Il peut s\'agir :\n\n- d\'une atteinte à l\'intégrité physique ou psychique ;\n\n- d\'une perte de chance de survie ou de guérison ;\n\n- d\'une perte de chance d\'avoir échappé à un risque qui s\'est finalement réalisé (en cas de défaut d\'information).\n\nLe dommage doit être actuel et certain. Il peut être futur dès lors qu\'il est certain (par exemple, la stérilité d\'une enfant du fait d\'une irradiation fautive).\n\nLe lien de causalité entre le fait générateur de responsabilité (en règle générale, la faute) et le dommage doit être certain. Il n\'a pas à être direct ni exclusif.\n\nParmi les éléments du triptyque fondant l\'engagement de la responsabilité, le lien de causalité est habituellement le plus difficile à prouver par le patient. En effet, il est souvent complexe de distinguer les conséquences de la faute de celles de l\'évolution spontanée de l\'état de santé pathologique. Le juge admet que l\'on puisse indemniser la perte de chance de survie ou de guérison. Dans ce cas, l\'indemnisation est accordée à proportion de la probabilité de survie ou de guérison perdue du fait de la faute.',
      lisa_metadata: {
        identifiant: 'OIC-005-01-A',
        rang: 'A',
        rubrique: 'Définition',
        intitule: 'Connaître la définition de la responsabilité sanction/indemnisation',
        item_parent: '',
        description: 'None',
        contenu: 'La responsabilité d\'un professionnel de santé...'
      }
    };

    console.log('📦 Mock MCQ created:', mockMCQ);

    this.currentMcq.set(mockMCQ);
    this.currentIndex.set(index);
    this.loading.set(false);

    console.log('✅ loadMCQ finished - loading:', this.loading(), 'currentMcq set:', !!this.currentMcq(), 'currentIndex:', this.currentIndex());
  }

  /**
   * Toggle le statut de validation d'un check (Section A ou B)
   * La checkbox permet à l'expert de valider le résultat du backend
   */
  toggleCheck(section: 'A' | 'B', checkIndex: number): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const checks = section === 'A' ? mcq.section_a_checks : mcq.section_b_checks;
    const check = checks[checkIndex];

    console.log(`🔄 toggleCheck - Section ${section}, Index ${checkIndex}, Current status: ${check.status}`);

    // Simple toggle: not_checked ↔ validated
    if (check.status === 'not_checked') {
      check.status = 'validated';
      check.confidence = 'high';
    } else {
      check.status = 'not_checked';
      check.confidence = null;
    }

    console.log(`✅ toggleCheck - New status: ${check.status}`);

    // Forcer la mise à jour du signal
    this.currentMcq.set({ ...mcq });
  }

  /**
   * Accepter le MCQ
   */
  onAccept(): void {
    if (confirm('Accepter ce MCQ ?')) {
      this.submitValidation('ACCEPT');
    }
  }

  /**
   * Rejeter le MCQ
   */
  onReject(): void {
    if (confirm('Rejeter ce MCQ ?')) {
      this.submitValidation('REJECT');
    }
  }

  /**
   * Soumettre la validation au backend
   */
  private submitValidation(decision: 'ACCEPT' | 'REJECT'): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const validationData = {
      section_a_checks: mcq.section_a_checks,
      section_b_checks: mcq.section_b_checks,
      human_decision: decision,
      human_feedback: this.validationForm.get('feedback')?.value || ''
    };

    console.log('📤 Soumission de la validation:', validationData);

    // Utiliser la nouvelle méthode qui gère localStorage + backend
    this.validationService.saveValidationToBackend(
      mcq.item_id,
      decision,
      validationData
    ).subscribe({
      next: (response) => {
        console.log('✅ Validation sauvegardée:', response);
        this.navigateAfterValidation();
      },
      error: (error) => {
        console.warn('⚠️ Erreur lors de la sauvegarde:', error);
        // Quand même naviguer car la validation est sauvegardée localement
        this.navigateAfterValidation();
      }
    });
  }

  /**
   * Navigation après validation
   */
  private navigateAfterValidation(): void {
    // Passer au MCQ suivant ou retourner au dashboard
    if (this.hasNext()) {
      this.onNext();
    } else {
      alert('Toutes les évaluations sont terminées !');
      this.mcqService.clearProgress();
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * MCQ précédent
   */
  onPrevious(): void {
    if (this.hasPrevious()) {
      this.loadMCQ(this.currentIndex() - 1);
    }
  }

  /**
   * MCQ suivant
   */
  onNext(): void {
    if (this.hasNext()) {
      this.loadMCQ(this.currentIndex() + 1);
    }
  }

  /**
   * Faire une pause et sauvegarder la progression
   */
  onPause(): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    // Sauvegarder la progression
    this.mcqService.saveProgress({
      current_index: this.currentIndex(),
      mcq_list: this.mcqList(),
      current_mcq_state: {
        mcq: mcq,
        feedback: this.validationForm.get('feedback')?.value || ''
      }
    });

    alert('✅ Votre progression a été sauvegardée. Vous pourrez reprendre plus tard.');
    this.router.navigate(['/dashboard']);
  }

  /**
   * Retour au dashboard
   */
  onBackToDashboard(): void {
    if (confirm('Quitter l\'évaluation et retourner au tableau de bord ?')) {
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * Vérifier si on peut aller au précédent
   */
  hasPrevious(): boolean {
    return this.currentIndex() > 0;
  }

  /**
   * Vérifier si on peut aller au suivant
   */
  hasNext(): boolean {
    return this.currentIndex() < this.mcqList().length - 1;
  }

  /**
   * Obtenir le statut du check (icône)
   */
  getCheckIcon(status: string): string {
    if (status === 'pass') return '✓';
    if (status === 'fail') return '✗';
    return '○';
  }

  /**
   * Obtenir la classe CSS du check
   */
  getCheckClass(status: string): string {
    if (status === 'pass') return 'pass';
    if (status === 'fail') return 'fail';
    return 'not-checked';
  }
}
