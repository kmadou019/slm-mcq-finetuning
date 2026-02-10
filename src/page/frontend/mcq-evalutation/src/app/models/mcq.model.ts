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
  status: 'not_checked' | 'pass' | 'fail';
  confidence: 'high' | 'medium' | 'low' | null;
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
