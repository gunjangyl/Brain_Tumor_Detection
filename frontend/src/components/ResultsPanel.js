import React from 'react';
import { Activity, ShieldAlert, CheckCircle, TrendingUp, AlertTriangle, FileText } from 'lucide-react';

const ResultsPanel = ({ data }) => {
    if (!data) {
        return (
            <div className="glass p-5 rounded-2xl flex-1 flex flex-col items-center justify-center text-slate-400">
                <Activity className="w-12 h-12 mb-3 opacity-50" />
                <p className="text-sm font-medium">Awaiting MRI scan input...</p>
            </div>
        );
    }

    const { predicted_class, confidence, probabilities } = data;
    const isHealthy = predicted_class === "No Tumor";

    return (
        <div className="glass p-5 rounded-2xl flex-1 flex flex-col overflow-y-auto">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center">
                <FileText className="w-4 h-4 mr-2" />
                2. Diagnostic Results
            </h3>

            {/* Main Verdict Card */}
            <div className={`p-4 rounded-xl border-l-4 shadow-sm mb-6 ${isHealthy ? 'bg-success/5 border-success' : 'bg-danger/5 border-danger'}`}>
                <div className="flex items-start justify-between">
                    <div>
                        <p className="text-xs uppercase font-bold text-slate-500 mb-1">Primary Detection</p>
                        <h2 className={`text-2xl font-black ${isHealthy ? 'text-success' : 'text-danger'}`}>
                            {predicted_class}
                        </h2>
                    </div>
                    {isHealthy ? (
                        <CheckCircle className="w-8 h-8 text-success opacity-80" />
                    ) : (
                        <ShieldAlert className="w-8 h-8 text-danger opacity-80" />
                    )}
                </div>
                <div className="mt-3 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-2 text-slate-500" />
                    <p className="text-sm font-medium text-slate-700">AI Confidence: <span className="font-bold">{confidence}</span></p>
                </div>
            </div>

            {/* Probability Distribution */}
            <h4 className="text-xs font-bold text-slate-500 uppercase mb-3 border-b border-slate-200 pb-2">Class Probability Distribution</h4>
            <div className="space-y-3 mb-6">
                {probabilities && Object.entries(probabilities).map(([className, score]) => {
                    const numScore = parseFloat(score);
                    const isHighest = className === predicted_class;

                    return (
                        <div key={className} className="relative">
                            <div className="flex justify-between text-xs mb-1 font-medium">
                                <span className={`${isHighest ? 'text-primary font-bold' : 'text-slate-600'}`}>{className}</span>
                                <span className="text-slate-500">{score}</span>
                            </div>
                            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all duration-1000 ${isHighest ? (isHealthy ? 'bg-success' : 'bg-danger') : 'bg-slate-300'}`}
                                    style={{ width: `${numScore}%` }}
                                ></div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Recommended Action */}
            <div className="mt-auto bg-slate-800 rounded-xl p-4 text-white shadow-lg">
                <h4 className="text-sm font-bold flex items-center mb-2">
                    <AlertTriangle className="w-4 h-4 mr-2 text-yellow-400" />
                    Recommended Action
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                    {isHealthy
                        ? "No morphological anomalies detected. Standard annual check-up recommended. Please verify visually with lead radiologist."
                        : `High probability of malignant/benign ${predicted_class} mass. Immediate biopsy and neurosurgical consultation required.`}
                </p>
            </div>

        </div>
    );
};

export default ResultsPanel;
