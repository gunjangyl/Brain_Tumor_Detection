import React, { useState } from 'react';
import DashboardLayout from './components/DashboardLayout';
import Uploader from './components/Uploader';
import ResultsPanel from './components/ResultsPanel';
import BrainModel3D from './components/BrainModel3D';

function App() {
  // Global React State holding the ML Pipeline prediction JSON from Flask
  const [predictionData, setPredictionData] = useState(null);

  return (
    <div className="min-h-screen bg-slate-50 w-full overflow-hidden flex items-center justify-center font-sans antialiased">
      {/* 
        The DashboardLayout acts as the master Grid/Flex skeleton. 
        It receives the raw components as children so it remains decoupled.
      */}
      <DashboardLayout activeComponent={setPredictionData} predictionData={predictionData}>

        {/* Child 1: Left Column Data Panel */}
        <div className="flex flex-col h-full space-y-6 w-full">
          <Uploader setPredictionData={setPredictionData} />
          <ResultsPanel data={predictionData} />
        </div>

        {/* Child 2: Right Column 3D Viewer */}
        <div className="h-full w-full relative">
          <BrainModel3D predictionData={predictionData} />
        </div>

      </DashboardLayout>
    </div>
  );
}

export default App;
