/**
 * Models for API requests and responses
 */

import { MCQCard } from './mcq.model';

export interface MCQListResponse {
  total: number;
  cards: MCQCard[];
}

export interface ValidationSubmission {
  mcq_id: string;
  decision: 'ACCEPT' | 'REJECT';
  feedback: string;
  validated_checks: Record<string, boolean>;
}

export interface ExportData {
  validations: ValidationSubmission[];
  export_date: string;
}

export interface CreateMCQDto {
  question: string;
  options: {
    A: string;
    B: string;
    C: string;
    D: string;
  };
  correct_option: string;
  source_material: string;
  generator_info: string;
}

export interface StatsResponse {
  total: number;
  accepted: number;
  rejected: number;
  pending: number;
}

export interface ApiError {
  message: string;
  status: number;
  details?: any;
}
