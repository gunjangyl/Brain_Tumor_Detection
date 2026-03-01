import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, useGLTF, Html, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// Fallback sphere geometry simulating a brain model if a custom GLTF/OBJ isn't provided
const FallbackBrain = ({ hasTumor }) => {
    const meshRef = useRef();

    // Idle pulsing and rotation animation
    useFrame((state, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.2;
            meshRef.current.rotation.x += Math.sin(state.clock.elapsedTime) * 0.002;

            // Heartbeat pulse if tumor is present
            if (hasTumor) {
                meshRef.current.scale.setScalar(
                    1 + Math.sin(state.clock.elapsedTime * 4) * 0.02
                );
            }
        }
    });

    return (
        <group>
            {/* Base Anatomical Mesh */}
            <mesh ref={meshRef} receiveShadow castShadow>
                <sphereGeometry args={[2, 64, 64]} />
                <meshPhysicalMaterial
                    color={hasTumor ? "#ffccd5" : "#dbeafe"}
                    transmission={0.5}
                    opacity={0.8}
                    metalness={0.1}
                    roughness={0.4}
                    ior={1.5}
                    thickness={2}
                    transparent
                />

                {/* Core Emissive Tumor Core */}
                {hasTumor && (
                    <mesh position={[0.5, 0.5, 1.2]}>
                        <sphereGeometry args={[0.4, 32, 32]} />
                        <meshStandardMaterial
                            color="#EF4444"
                            emissive="#ff0000"
                            emissiveIntensity={2.5}
                        />

                        {/* HTML Tooltip mapped directly to the 3D coordinate */}
                        <Html position={[0.6, 0.6, 0]}>
                            <div className="bg-slate-900/80 backdrop-blur-md p-3 rounded-xl border border-danger shadow-xl text-white w-48 -translate-y-1/2">
                                <p className="font-bold text-danger text-sm flex items-center mb-1">
                                    <span className="w-2 h-2 rounded-full bg-danger animate-pulse mr-2"></span>
                                    Malignant Mass
                                </p>
                                <p className="text-xs text-slate-300">Volume: Est. 14.2cm³</p>
                                <p className="text-xs text-slate-300">Region: Parietal Lobe</p>
                            </div>
                        </Html>
                    </mesh>
                )}
            </mesh>
        </group>
    );
};

const BrainModel3D = ({ predictionData }) => {
    const [hasTumor, setHasTumor] = useState(false);

    // Sync state when backend returns data
    useEffect(() => {
        if (predictionData && predictionData.predicted_class !== "No Tumor") {
            setHasTumor(true);
        } else {
            setHasTumor(false);
        }
    }, [predictionData]);

    return (
        <div className="w-full h-full absolute inset-0 rounded-3xl overflow-hidden glass z-0">

            {/* Diagnostic Overlay HUD */}
            <div className="absolute top-6 left-6 z-10">
                <h2 className="text-3xl font-black text-slate-800 drop-shadow-sm tracking-tight flex items-center">
                    Interactive 3D Viewer
                </h2>
                <p className={`text-sm font-semibold mt-2 px-3 py-1 rounded-full w-max shadow-sm 
           ${predictionData ? (hasTumor ? 'bg-danger/20 text-danger border border-danger/30' : 'bg-success/20 text-success border border-success/30')
                        : 'bg-slate-200 text-slate-500'}`}>
                    {predictionData ? (hasTumor ? 'ANOMALY DETECTED' : 'HEALTHY SCAN') : 'AWAITING SCAN...'}
                </p>
            </div>

            <Canvas shadows camera={{ position: [0, 0, 8], fov: 45 }}>
                <ambientLight intensity={0.4} />
                <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
                <pointLight position={[-10, -10, -10]} intensity={0.5} />

                {/* Soft HDRI medical lighting */}
                <Environment preset="studio" />

                {/* 3D Anatomical Fallback (Since true organic Models require external `.gltf` hosting) */}
                <FallbackBrain hasTumor={hasTumor} />

                {/* Soft ground reflection */}
                <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={20} blur={2} far={4} />

                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    minDistance={3}
                    maxDistance={12}
                    autoRotate={!hasTumor} /* Stop rotating if they need to inspect the tumor */
                    autoRotateSpeed={0.5}
                />
            </Canvas>
        </div>
    );
};

export default BrainModel3D;
