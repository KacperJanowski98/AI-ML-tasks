// Medical note sections as defined in original Python code
export type MedicalSection = 'Wywiad' | 'Badanie' | 'Diagnoza' | 'Zalecenia' | 'Kontekst';

// Medical note with sections and content
export interface MedicalNote {
  sections: Record<MedicalSection, string[]>;
}

// Section detection keywords (moved from Python regex to TypeScript)
export const SECTION_KEYWORDS: Record<MedicalSection, string[]> = {
  Badanie: ['bada', 'osłuch', 'palp', 'temp'],
  Diagnoza: ['rozpozn', 'diagnoz', 'podejrz'],
  Zalecenia: ['zalec', 'przepis', 'kontrol'],
  Kontekst: ['dom', 'rodzin', 'środowisk', 'koleżank'],
  Wywiad: [] // Default section
};