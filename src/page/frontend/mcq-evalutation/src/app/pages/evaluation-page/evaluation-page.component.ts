import { Component, inject, OnInit, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { McqService, AssignedMCQ } from '../../services/mcq.service';
import { ValidationService } from '../../services/validation.service';
import { MCQCard, SectionCheck } from '../../models';
import { Subscription } from 'rxjs';

interface SessionSummary {
  total: number;
  accepted: number;
  rejected: number;
}

/**
 * EvaluationPage Component - Page d'evaluation des MCQ
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
  private readonly route = inject(ActivatedRoute);
  private routerSubscription?: Subscription;

  loading = signal(true);
  currentIndex = signal(0);
  currentMcq = signal<MCQCard | null>(null);
  mcqList = signal<AssignedMCQ[]>([]);
  sessionSummary = signal<SessionSummary | null>(null);

  // Review mode (Feature 4)
  reviewMode = signal(false);
  previousDecision = signal<'ACCEPT' | 'REJECT' | null>(null);

  // Session counters (Feature 3)
  private sessionAccepted = 0;
  private sessionRejected = 0;

  validationForm: FormGroup;

  constructor() {
    this.validationForm = this.fb.group({
      feedback: ['']
    });
  }

  ngOnInit(): void {
    // Feature 4 — Check for review mode query params
    const mcqId = this.route.snapshot.queryParamMap.get('mcq_id');
    const model = this.route.snapshot.queryParamMap.get('model');

    if (mcqId && model) {
      this.reviewMode.set(true);
      this.loadReviewMode(mcqId, model);
    } else {
      this.initializeEvaluation();
    }
  }

  ngOnDestroy(): void {
    this.routerSubscription?.unsubscribe();
  }

  // Feature 4 — Load a specific MCQ for review
  private loadReviewMode(mcqId: string, model: string): void {
    this.loading.set(true);
    this.mcqService.getMcqById(mcqId, model).subscribe({
      next: (mcq) => {
        this.currentMcq.set(mcq);
        this.mcqList.set([{ mcq_id: mcqId, model }]);
        this.currentIndex.set(0);
        this.loading.set(false);

        // Pre-fill feedback from history
        this.mcqService.getValidationHistory(200).subscribe({
          next: (history) => {
            const entry = history.find((h: any) => h.mcq_id === mcqId && h.model === model);
            if (entry) {
              this.previousDecision.set(entry.decision as 'ACCEPT' | 'REJECT');
              this.validationForm.patchValue({ feedback: entry.human_feedback || '' });
            }
          },
          error: () => {}
        });
      },
      error: () => { this.loading.set(false); }
    });
  }

  private initializeEvaluation(): void {
    this.reviewMode.set(false);
    this.previousDecision.set(null);
    this.sessionSummary.set(null);
    this.sessionAccepted = 0;
    this.sessionRejected = 0;
    this.loading.set(true);
    this.validationService.syncWithBackend().subscribe({
      next: () => this.loadAssignedMCQs(),
      error: () => this.loadAssignedMCQs()
    });
  }

  /**
   * Charger les MCQ assignes depuis le backend, en filtrant ceux deja evalues
   */
  private loadAssignedMCQs(): void {
    this.loading.set(true);

    this.mcqService.getAssignedMcqs().subscribe({
      next: (response) => {
        const validations = this.validationService.getAllValidations();

        // Filtrer en utilisant la cle composite mcq_id::model
        const pendingMcqs = response.assignments.filter(a => {
          const key = this.validationService.makeKey(a.mcq_id, a.model);
          return !validations[key];
        });

        console.log(`MCQs assignes: ${response.assignments.length}, en attente: ${pendingMcqs.length}`);

        this.mcqList.set(pendingMcqs);

        if (pendingMcqs.length > 0) {
          const savedProgress = this.mcqService.loadProgress();
          let startIndex = 0;

          if (savedProgress && savedProgress.mcq_list.length > 0) {
            const lastItem = savedProgress.mcq_list[savedProgress.current_index];
            if (lastItem) {
              const resumeIndex = pendingMcqs.findIndex(
                m => m.mcq_id === lastItem.mcq_id && m.model === lastItem.model
              );
              if (resumeIndex >= 0) {
                startIndex = resumeIndex;
              }
            }
            this.mcqService.clearProgress();
          }

          this.loadMCQ(startIndex);
        } else {
          this.loading.set(false);
          this.mcqService.clearProgress();
          alert('Toutes les evaluations sont terminees ! Vous pouvez demander de nouveaux QCMs depuis le dashboard.');
          this.router.navigate(['/dashboard']);
        }
      },
      error: (error) => {
        console.error('Erreur backend:', error);
        this.loading.set(false);
      }
    });
  }

  /**
   * Charger un MCQ specifique depuis le backend
   */
  private loadMCQ(index: number): void {
    this.loading.set(true);
    const item = this.mcqList()[index];

    this.mcqService.getMcqById(item.mcq_id, item.model).subscribe({
      next: (mcq) => {
        this.currentMcq.set(mcq);
        this.currentIndex.set(index);
        this.loading.set(false);
        this.validationForm.reset();
      },
      error: (error) => {
        console.error('Erreur chargement MCQ:', error);
        // MCQ introuvable (fichier supprimé) : passer au suivant
        if (error.status === 404 && index + 1 < this.mcqList().length) {
          this.loadMCQ(index + 1);
        } else {
          this.loading.set(false);
        }
      }
    });
  }

  toggleCheck(section: 'A' | 'B', checkIndex: number): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const checks = section === 'A' ? mcq.section_a_checks : mcq.section_b_checks;
    const check = checks[checkIndex];

    if (check.status === 'not_checked') {
      check.status = 'validated';
      check.confidence = 'high';
    } else {
      check.status = 'not_checked';
      check.confidence = null;
    }

    this.currentMcq.set({ ...mcq });
  }

  onAccept(): void {
    if (confirm('Accepter ce QCM ?')) {
      this.submitValidation('ACCEPT');
    }
  }

  onReject(): void {
    if (confirm('Rejeter ce QCM ?')) {
      this.submitValidation('REJECT');
    }
  }

  /**
   * Soumettre la validation au backend
   */
  private submitValidation(decision: 'ACCEPT' | 'REJECT'): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    const currentItem = this.mcqList()[this.currentIndex()];

    const validationData = {
      section_a_checks: mcq.section_a_checks,
      section_b_checks: mcq.section_b_checks,
      human_decision: decision,
      human_feedback: this.validationForm.get('feedback')?.value || '',
      mcq_data: {
        item_id: mcq.item_id,
        mcq_question: mcq.mcq_question,
        options: mcq.options,
        correct_option: mcq.correct_option,
        source_material: mcq.source_material,
        generator_info: mcq.generator_info,
        final_decision: mcq.final_decision
      },
      content_raw: (mcq as any).lisa_texte_brut || ''
    };

    // Feature 3 — count session decisions (not in review mode)
    if (!this.reviewMode()) {
      if (decision === 'ACCEPT') this.sessionAccepted++;
      else this.sessionRejected++;
    }

    this.validationService.saveValidationToBackend(
      mcq.item_id,
      currentItem.model,
      decision,
      validationData
    ).subscribe({
      next: () => this.navigateAfterValidation(),
      error: () => this.navigateAfterValidation()
    });
  }

  private navigateAfterValidation(): void {
    // Feature 4 — review mode: go back to history
    if (this.reviewMode()) {
      this.router.navigate(['/history']);
      return;
    }

    if (this.hasNext()) {
      this.onNext();
    } else {
      // Feature 3 — show session summary overlay instead of alert
      this.mcqService.clearProgress();
      this.sessionSummary.set({
        total: this.sessionAccepted + this.sessionRejected,
        accepted: this.sessionAccepted,
        rejected: this.sessionRejected,
      });
    }
  }

  onGoToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  onGoToHistory(): void {
    this.router.navigate(['/history']);
  }

  onPrevious(): void {
    if (this.hasPrevious()) {
      this.loadMCQ(this.currentIndex() - 1);
    }
  }

  onNext(): void {
    if (this.hasNext()) {
      this.loadMCQ(this.currentIndex() + 1);
    }
  }

  onPause(): void {
    const mcq = this.currentMcq();
    if (!mcq) return;

    this.mcqService.saveProgress({
      current_index: this.currentIndex(),
      mcq_list: this.mcqList(),
      current_mcq_state: {
        mcq: mcq,
        feedback: this.validationForm.get('feedback')?.value || ''
      }
    });

    alert('Votre progression a ete sauvegardee.');
    this.router.navigate(['/dashboard']);
  }

  onBackToDashboard(): void {
    if (confirm('Quitter l\'evaluation et retourner au tableau de bord ?')) {
      this.router.navigate(['/dashboard']);
    }
  }

  hasPrevious(): boolean {
    return this.currentIndex() > 0;
  }

  hasNext(): boolean {
    return this.currentIndex() < this.mcqList().length - 1;
  }

  getCheckIcon(status: string): string {
    if (status === 'pass') return '\u2713';
    if (status === 'fail') return '\u2717';
    return '\u25CB';
  }

  getCheckClass(status: string): string {
    if (status === 'pass') return 'pass';
    if (status === 'fail') return 'fail';
    return 'not-checked';
  }
}
