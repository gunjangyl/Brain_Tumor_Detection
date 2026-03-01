import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, Loader2, FileImage } from 'lucide-react';
import axios from 'axios';

const Uploader = ({ setPredictionData }) => {
    const [dragActive, setDragActive] = useState(false);
    const [loading, setLoading] = useState(false);
    const [fileName, setFileName] = useState('');
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const processFile = async (file) => {
        if (!file) return;
        setFileName(file.name);
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Connects to Flask backend API - Using 127.0.0.1 for better IP resolution consistency
            const response = await axios.post('http://127.0.0.1:4555/api/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            // Delay slightly for presentation smoothness
            setTimeout(() => {
                setPredictionData(response.data);
                setLoading(false);
            }, 800);

        } catch (error) {
            console.error("Diagnostic Error:", error);
            const errorMsg = error.response?.data?.error || error.message;
            alert(`AI Diagnostics Server Error: ${errorMsg}. Please ensure Flask app is running on port 4555.`);
            setLoading(false);
            setPredictionData(null);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            processFile(e.target.files[0]);
        }
    };

    return (
        <div className="glass p-5 rounded-2xl w-full">
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">
                1. MRI Scan Input
            </h3>

            <div
                className={`relative border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all duration-300
          ${dragActive ? 'border-primary bg-primary/5 shadow-inner' : 'border-slate-300 hover:border-primary/50 hover:bg-slate-50'}
          ${loading ? 'opacity-80 pointer-events-none' : 'cursor-pointer'}
        `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg, image/png, image/jpg"
                    className="hidden"
                    onChange={handleChange}
                />

                {loading ? (
                    <div className="flex flex-col items-center animate-pulse text-primary">
                        <Loader2 className="w-12 h-12 animate-spin mb-3" />
                        <p className="font-medium">AI Analyzing Scan...</p>
                        <p className="text-xs text-slate-500 mt-1">{fileName}</p>
                    </div>
                ) : fileName ? (
                    <div className="flex flex-col items-center text-success">
                        <CheckCircle2 className="w-12 h-12 mb-3 drop-shadow-sm" />
                        <p className="font-bold text-slate-700">Scan Uploaded</p>
                        <div className="flex items-center mt-2 text-xs text-slate-500 bg-white px-3 py-1 rounded-full shadow-sm">
                            <FileImage className="w-3 h-3 mr-1" />
                            {fileName}
                        </div>
                        <p className="text-xs text-primary mt-4 font-semibold hover:underline">Click to process another</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center text-slate-400">
                        <div className="bg-white p-3 rounded-full shadow-sm mb-3 group-hover:shadow-md transition">
                            <UploadCloud className="w-8 h-8 text-primary" />
                        </div>
                        <p className="font-semibold text-slate-600 mb-1">Drag & Drop MRI</p>
                        <p className="text-xs">or click to browse local files</p>
                        <p className="text-[10px] uppercase font-bold tracking-widest mt-4 opacity-50">JPEG • PNG</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Uploader;
