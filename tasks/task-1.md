# Task 1: Smart Nature Observer

## Objective
Create a prototype video analytics system for natural environments that can detect and label both static (e.g., mountains, trees) and dynamic objects (e.g., animals, birds). The system should provide an intuitive interface for reviewing and refining AI-generated segmentations through a modern, responsive canvas tool.

## Technical Requirements

1. Frontend Stack:
   ```typescript
   // Core technologies
   - React + TypeScript
   - Canvas/WebGL for visualization
   - Web Workers for processing
   ```

2. Video Processing:
   ```typescript
   interface VideoProcessor {
     // Configuration
     config: {
       processingResolution: {
         width: number;  // suggest 720p/360p for processing
         height: number;
       };
       outputResolution: {
         width: number;  // original video resolution
         height: number;
       };
       model: {
         type: 'SAM2' | 'other';  // SAM 2.1 recommended, but open to alternatives
         configuration: Record<string, any>;
       };
     };

     // Processing methods
     processVideo: (file: File) => Promise<ProcessedVideoData>;
     extractFrame: (timestamp: number) => Promise<FrameData>;
     updateSegmentation: (frameId: string, updates: SegmentationUpdate) => Promise<void>;
   }

   interface FrameData {
     id: string;
     timestamp: number;
     segmentation: {
       masks: Mask[];
       labels: Label[];
       confidence: number[];
     };
     thumbnail: string;  // Base64 for filmroll
   }
   ```

3. Interactive Visualization:
   ```typescript
   interface CanvasEditor {
     // Canvas setup with resolution handling
     initialize: (container: HTMLElement) => void;
     setResolution: (width: number, height: number) => void;
     
     // Frame navigation (5-10 fps precision)
     seekTo: (timestamp: number) => Promise<void>;
     getCurrentFrame: () => FrameData;
     
     // Interaction handlers
     onPointerDown?: (event: PointerEvent) => void;
     onPointerMove?: (event: PointerEvent) => void;
     onPointerUp?: (event: PointerEvent) => void;
     
     // Mask manipulation
     updateMask: (maskId: string, updates: Partial<Mask>) => void;
     createMask: (points: Point[]) => Promise<string>;
     deleteMask: (maskId: string) => void;
   }
   ```

4. Resolution Management:
   ```typescript
   class ResolutionManager {
     constructor(originalVideo: HTMLVideoElement) {
       this.originalResolution = {
         width: originalVideo.videoWidth,
         height: originalVideo.videoHeight
       };
     }

     getProcessingResolution() {
       // Implement smart downscaling logic
       // Suggest 720p or 360p based on original size
       return this.calculateOptimalProcessingSize();
     }

     scaleSegmentationToOriginal(mask: Mask): Mask {
       // Scale mask coordinates back to original resolution
       // Important: Handle normalization correctly here
       return this.normalizeAndScaleMask(mask);
     }
   }
   ```

5. Performance Optimization:
   ```typescript
   // Example implementation suggestions
   interface PerformanceOptimizations {
     // Frame buffering
     frameBuffer: {
       ahead: number;  // Number of frames to pre-process
       behind: number; // Number of frames to keep in memory
     };

     // Render optimization
     canvasLayers: {
       background: HTMLCanvasElement;  // Original video
       masks: HTMLCanvasElement;       // Segmentation overlay
       interaction: HTMLCanvasElement; // Active editing layer
     };

     // WebGL acceleration
     glContext?: WebGLRenderingContext;
     shaders?: {
       maskBlending: WebGLProgram;
       colorization: WebGLProgram;
     };
   }
   ```

## Expected Deliverables

1. Working Prototype:
   - Video upload and processing interface
   - Frame-by-frame review capability (5-10 fps precision)
   - Interactive segmentation adjustment tools
   - Export functionality for processed results

2. Documentation:
   - Chosen model explanation and justification
   - Resolution management strategy
   - Performance optimization techniques
   - API documentation

3. Code Structure:
   ```
   src/
   ├── components/
   │   ├── VideoProcessor/
   │   │   ├── index.tsx
   │   │   ├── ResolutionManager.ts
   │   │   └── FrameProcessor.ts
   │   ├── Canvas/
   │   │   ├── Editor.tsx
   │   │   ├── Tools.tsx
   │   │   └── Filmroll.tsx
   │   └── UI/
   │       ├── Controls.tsx
   │       └── Timeline.tsx
   ├── hooks/
   │   ├── useVideoProcessing.ts
   │   ├── useCanvasEditor.ts
   │   └── useSegmentation.ts
   ├── lib/
   │   ├── models.ts
   │   ├── videoUtils.ts
   │   └── canvasUtils.ts
   └── types/
       └── index.ts
   ```

## Evaluation Criteria

1. Technical Implementation:
   - Proper resolution management
   - Efficient canvas operations
   - Smooth video timeline navigation
   - Effective use of Web Workers

2. Code Quality:
   - TypeScript usage
   - Performance considerations
   - Error handling
   - Memory management

3. User Experience:
   - Tool responsiveness
   - Interface intuitiveness
   - Visual feedback
   - Performance under load

4. Documentation:
   - Implementation decisions
   - Performance considerations
   - Setup instructions
   - Future improvements
