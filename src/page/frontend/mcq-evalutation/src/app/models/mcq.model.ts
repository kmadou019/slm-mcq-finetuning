/**
 * Models for MCQ (Multiple Choice Questions) and LISA Sheets
 */

export interface MCQOption {
  A: string;
  B: string;
  C: string;
  D: string;
  [key: string]: string; // Allow dynamic access
}

export interface SectionCheck {
  check_id: string;
  description: string;
  result: 'PASS' | 'WARN';  // Result from backend (read-only)
  status: 'not_checked' | 'validated' | 'rejected';  // Expert validation status
  confidence: 'high' | 'medium' | 'low' | null;
  threshold?: string;  // For Section A
  score?: string;      // For Section B
  notes?: string;      // Notes about the check
}

export interface LISAMetadata {
  identifiant: string;
  rang: string;
  rubrique: string;
  intitule: string;
  item_parent: string;
  description: string;
  contenu: string;
}

export interface MCQCard {
  item_id: string;
  source_material: string;
  generator_info: string;
  output_format: string;
  mcq_question: string;
  options: MCQOption;
  correct_option: string;
  section_a_checks: SectionCheck[];
  section_b_checks: SectionCheck[];
  decision_policy: string;
  final_decision: 'ACCEPT' | 'REVISE';
  audit_trail: string;
  lisa_texte_brut: string;
  lisa_metadata?: LISAMetadata;  // Optional LISA metadata
}

export interface LISASheet {
  identifiant: string;
  rang: string;
  intitule: string;
  description: string;
  rubrique: string;
  item_parent: string;
  contenu: string;
}

export interface ValidationData {
  index: number;
  human_decision: 'ACCEPT' | 'REJECT' | null;
  human_feedback: string;
  validated_fields: Record<string, boolean>;
  timestamp: string;
}
