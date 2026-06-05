import {
  Component, signal, OnInit, OnDestroy, AfterViewInit,
  inject, ViewChild, ElementRef, effect,
} from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { interval, Subscription } from 'rxjs';
import { switchMap, takeWhile } from 'rxjs/operators';
import {
  Chart, CategoryScale, LinearScale, BarElement,
  BarController, Title, Tooltip, Legend,
} from 'chart.js';
import {
  AnswerabilityService,
  AnswerabilityRequest,
  AnswerabilityJob,
  AnswerabilityResult,
  ModelCardInfo,
} from '../../services/answerability.service';
import { AuthService } from '../../services/auth.service';

Chart.register(CategoryScale, LinearScale, BarElement, BarController, Title, Tooltip, Legend);

// ---- Presets & constants ----

interface DatasetPreset {
  label: string;
  dataset_name: string;
  dataset_split: string;
  preprocess_mode: string;
  question_col: string;
  option_a_col: string;
  option_b_col: string;
  option_c_col: string;
  option_d_col: string;
  correct_col: string;
  internal: boolean;
}

const MEDMCQA_COLS = {
  question_col: 'question',
  option_a_col: 'opa',
  option_b_col: 'opb',
  option_c_col: 'opc',
  option_d_col: 'opd',
  correct_col: 'cop',
};

const DATASET_PRESETS: DatasetPreset[] = [
  {
    label: 'MedMCQA (stratifié)',
    dataset_name: 'openlifescienceai/medmcqa',
    dataset_split: 'train',
    preprocess_mode: 'medmcqa_stratified',
    ...MEDMCQA_COLS,
    internal: false,
  },
  {
    label: 'UNESS',
    dataset_name: 'uness',
    dataset_split: '',
    preprocess_mode: 'none',
    ...MEDMCQA_COLS,
    internal: true,
  },
];

const DATASET_LABELS: Record<string, string> = {
  'uness': 'UNESS',
  'openlifescienceai/medmcqa': 'MedMCQA',
};

// Consistent color palette — one color per model (by insertion order)
const BAR_COLORS = [
  '#f59e0b', '#ef4444', '#06b6d4', '#84cc16',
  '#8b5cf6', '#ec4899', '#10b981', '#3b82f6',
];

@Component({
  selector: 'app-answerability-page',
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe],
  templateUrl: './answerability-page.component.html',
  styleUrl: './answerability-page.component.scss',
})
export class AnswerabilityPageComponent implements OnInit, AfterViewInit, OnDestroy {
  private service = inject(AnswerabilityService);
  private authService = inject(AuthService);
  private router = inject(Router);

  @ViewChild('chartCanvas') chartCanvasRef!: ElementRef<HTMLCanvasElement>;

  // --- UI state ---
  ollamaModels       = signal<string[]>([]);
  results            = signal<AnswerabilityResult[]>([]);
  currentJob         = signal<AnswerabilityJob | null>(null);
  isRunning          = signal(false);
  errorMsg           = signal<string | null>(null);
  modelCard          = signal<ModelCardInfo | null>(null);
  modelCardLoading   = signal(false);
  modelCardError     = signal<string | null>(null);

  readonly sectionLabels: Record<string, string> = {
    intended_use:           'Usage prévu',
    factors:                'Facteurs',
    training_data:          "Données d'entraînement",
    evaluation_data:        "Données d'évaluation",
    quantitative_analyses:  'Analyses quantitatives',
    ethical_considerations: 'Considérations éthiques',
    caveats:                'Mises en garde',
  };

  get modelCardSections(): Array<{ key: string; label: string; content: string }> {
    const sections = this.modelCard()?.sections ?? {};
    return Object.entries(sections)
      .filter(([, v]) => typeof v === 'string' && v.trim().length > 0)
      .map(([k, v]) => ({
        key:     k,
        label:   this.sectionLabels[k] ?? k,
        content: v as string,
      }));
  }

  readonly hfModels = [
    { label: 'Qwen3-0.6B', value: 'Qwen/Qwen3-0.6B' },
    { label: 'Llama 3.1 8B Instruct', value: 'meta-llama/Llama-3.1-8B-Instruct' },
    { label: 'MedGemma 27B', value: 'google/medgemma-27b-it' },
    { label: 'Gemma 4 31B', value: 'google/gemma-4-31B-it' },
  ];
  hfSelectValue = '';
  hfCustomInput = signal(false);
  hfCustomText = '';

  onHfSelectChange(val: string): void {
    if (val === '__custom__') {
      this.hfCustomInput.set(true);
      this.hfCustomText = '';
      this.form.model_name = '';
    } else {
      this.hfCustomInput.set(false);
      this.form.model_name = val;
    }
  }

  readonly presets = DATASET_PRESETS;

  // --- Form ---
  form: AnswerabilityRequest = {
    model_type: 'ollama',
    model_name: '',
    dataset_name: 'openlifescienceai/medmcqa',
    dataset_split: 'train',
    preprocess_mode: 'medmcqa_stratified',
    ...MEDMCQA_COLS,
    sample_size: null,
  };

  private pollSub: Subscription | null = null;
  private currentJobId: string | null = null;
  private chart: Chart | null = null;
  private viewReady = false;

  constructor() {
    // Re-render chart whenever results change
    effect(() => {
      const r = this.results();
      if (this.viewReady) this.updateChart(r);
    });
  }

  ngOnInit(): void {
    this.service.getOllamaModels().subscribe({
      next: (res) => this.ollamaModels.set(res.models),
      error: () => this.ollamaModels.set([]),
    });
    this.loadResults();
  }

  ngAfterViewInit(): void {
    this.viewReady = true;
    this.updateChart(this.results());
  }

  ngOnDestroy(): void {
    this.pollSub?.unsubscribe();
    this.chart?.destroy();
  }

  loadResults(): void {
    this.service.getResults().subscribe({
      next: (res) => this.results.set(res.results),
      error: () => {},
    });
  }

  onHfModelBlur(): void {
    const modelId = this.form.model_name.trim();
    if (!modelId) return;

    this.modelCard.set(null);
    this.modelCardError.set(null);
    this.modelCardLoading.set(true);

    this.service.getModelCard(modelId).subscribe({
      next: (card) => {
        this.modelCard.set(card);
        this.modelCardLoading.set(false);
      },
      error: (err) => {
        this.modelCardError.set(
          err?.error?.detail ?? 'Impossible de récupérer la model card.'
        );
        this.modelCardLoading.set(false);
      },
    });
  }

  applyPreset(preset: DatasetPreset): void {
    this.form.dataset_name    = preset.dataset_name;
    this.form.dataset_split   = preset.dataset_split;
    this.form.preprocess_mode = preset.preprocess_mode;
    this.form.question_col    = preset.question_col;
    this.form.option_a_col    = preset.option_a_col;
    this.form.option_b_col    = preset.option_b_col;
    this.form.option_c_col    = preset.option_c_col;
    this.form.option_d_col    = preset.option_d_col;
    this.form.correct_col     = preset.correct_col;
  }

  get isInternalDataset(): boolean {
    return this.form.dataset_name === 'uness';
  }

  launch(): void {
    this.errorMsg.set(null);
    if (!this.form.model_name.trim()) {
      this.errorMsg.set('Veuillez renseigner un nom de modèle.');
      return;
    }
    if (!this.form.dataset_name.trim()) {
      this.errorMsg.set('Veuillez renseigner un dataset.');
      return;
    }

    this.isRunning.set(true);
    this.currentJob.set(null);
    this.currentJobId = null;

    this.service.runEvaluation(this.form).subscribe({
      next: ({ job_id }) => { this.currentJobId = job_id; this.startPolling(job_id); },
      error: (err) => {
        this.isRunning.set(false);
        this.errorMsg.set(err?.error?.detail ?? 'Erreur lors du lancement.');
      },
    });
  }

  private startPolling(jobId: string): void {
    this.pollSub?.unsubscribe();
    this.pollSub = interval(2000)
      .pipe(
        switchMap(() => this.service.getJobStatus(jobId)),
        takeWhile(
          (job) => job.status !== 'done' && job.status !== 'error' && job.status !== 'cancelled',
          true
        )
      )
      .subscribe({
        next: (job) => {
          this.currentJob.set(job);
          if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled') {
            this.isRunning.set(false);
            this.loadResults();
          }
        },
        error: () => {
          this.isRunning.set(false);
          this.errorMsg.set('Erreur lors du suivi du job.');
        },
      });
  }

  get progressPercent(): number {
    const job = this.currentJob();
    if (!job || job.total === 0) return 0;
    return Math.round((job.progress / job.total) * 100);
  }

  get statusLabel(): string {
    const job = this.currentJob();
    if (!job) return '';
    switch (job.status) {
      case 'loading_dataset': return 'Chargement du dataset...';
      case 'loading_model':   return 'Chargement du modèle (peut prendre plusieurs minutes)...';
      case 'running':         return `Évaluation : ${job.progress} / ${job.total} questions`;
      case 'done':            return `Terminé — ${job.accuracy?.toFixed(2)}% de bonnes réponses`;
      case 'error':           return `Erreur : ${job.error}`;
      case 'cancelled':       return `Annulé après ${job.progress} / ${job.total} questions`;
      default:                return 'Démarrage...';
    }
  }

  cancelJob(): void {
    if (!this.currentJobId) return;
    this.service.cancelJob(this.currentJobId).subscribe({
      error: () => {},
    });
  }

  // ---- Chart ----

  private getDatasetLabel(name: string): string {
    return DATASET_LABELS[name] ?? name.split('/').pop() ?? name;
  }

  private updateChart(results: AnswerabilityResult[]): void {
    const canvas = this.chartCanvasRef?.nativeElement;
    if (!canvas) return;

    // Collect ordered unique datasets and models
    const datasetNames = [...new Set(results.map(r => r.dataset_name))];
    const modelNames   = [...new Set(results.map(r => r.model_name))];

    const datasetLabels = datasetNames.map(n => this.getDatasetLabel(n));

    // One Chart.js dataset per model
    const datasets = modelNames.map((model, i) => ({
      label: model,
      data: datasetNames.map(ds => {
        const r = results.find(x => x.model_name === model && x.dataset_name === ds);
        return r ? r.accuracy : null;
      }),
      backgroundColor: BAR_COLORS[i % BAR_COLORS.length] + 'cc', // slight transparency
      borderColor:     BAR_COLORS[i % BAR_COLORS.length],
      borderWidth: 1,
      borderRadius: 4,
    }));

    const minVal = results.length > 0
      ? Math.max(0, Math.floor(Math.min(...results.map(r => r.accuracy)) / 10) * 10 - 10)
      : 0;

    if (this.chart) {
      this.chart.data.labels = datasetLabels;
      this.chart.data.datasets = datasets;
      (this.chart.options.scales!['y'] as any).min = minVal;
      this.chart.update();
      return;
    }

    this.chart = new Chart(canvas, {
      type: 'bar',
      data: { labels: datasetLabels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { family: "'Segoe UI', sans-serif", size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "'Segoe UI', sans-serif", size: 12 } },
          },
          y: {
            min: minVal,
            max: 100,
            title: {
              display: true,
              text: 'Answerability (%)',
              font: { family: "'Segoe UI', sans-serif", size: 12 },
            },
            ticks: {
              callback: (v) => `${v}%`,
              font: { family: "'Segoe UI', sans-serif", size: 11 },
            },
          },
        },
      },
    });
  }

  onLogout(): void {
    this.authService.logout().subscribe();
    this.router.navigate(['/login']);
  }

  goToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }
}
