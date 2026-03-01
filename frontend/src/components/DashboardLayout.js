import React from 'react';
import { Activity, Menu, Bell, User, LayoutDashboard, Brain, FolderArchive, Settings } from 'lucide-react';
import Uploader from './Uploader';

const DashboardLayout = ({ children, activeComponent, predictionData }) => {
    return (
        <div className="flex h-screen w-full bg-slate-50/50 overflow-hidden text-slate-800 font-sans">

            {/* Sidebar - Glassmorphism */}
            <aside className="w-64 flex-shrink-0 glass m-3 rounded-2xl flex flex-col justify-between overflow-hidden shadow-xl border-white/40">
                <div>
                    <div className="h-20 flex items-center justify-center border-b border-slate-200/50 mx-4">
                        <Activity className="w-8 h-8 text-primary mr-2" />
                        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
                            NeuroScan AI
                        </span>
                    </div>

                    <nav className="p-4 space-y-2 mt-4">
                        <NavItem icon={<LayoutDashboard />} label="Diagnostic Panel" active />
                        <NavItem icon={<Brain />} label="3D Viewer" />
                        <NavItem icon={<FolderArchive />} label="Patient Records" />
                        <NavItem icon={<Settings />} label="AI Configuration" />
                    </nav>
                </div>

                <div className="p-4 border-t border-slate-200/50 m-4">
                    <div className="flex items-center space-x-3 p-2 rounded-xl hover:bg-white/50 cursor-pointer transition">
                        <div className="bg-primary/20 p-2 rounded-full">
                            <User className="text-primary w-5 h-5" />
                        </div>
                        <div>
                            <p className="font-semibold text-sm">Dr. Smith</p>
                            <p className="text-xs text-slate-500">Neurology Dept</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col h-full overflow-hidden relative">

                {/* Top Header navbar */}
                <header className="h-20 glass m-3 rounded-2xl flex items-center justify-between px-6 shadow-md z-10">
                    <div className="flex items-center text-slate-500">
                        <Menu className="w-6 h-6 mr-4 cursor-pointer hover:text-primary transition" />
                        <h2 className="font-semibold text-lg tracking-wide">Patient Diagnostics Workspace</h2>
                    </div>
                    <div className="flex items-center space-x-4">
                        <span className="flex h-3 w-3 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-success"></span>
                        </span>
                        <span className="text-sm font-medium text-slate-600">AI Server Online</span>
                        <div className="w-px h-6 bg-slate-200 mx-2"></div>
                        <Bell className="w-5 h-5 text-slate-500 hover:text-primary cursor-pointer transition" />
                    </div>
                </header>

                {/* Scrollable Workspace */}
                <div className="flex-1 overflow-y-auto w-full max-w-7xl mx-auto p-4 flex gap-6 pb-20">

                    {/* Left Column (Upload & Analysis Panel) */}
                    <div className="w-1/3 flex flex-col space-y-6">
                        <Uploader setPredictionData={activeComponent} />
                        {/* Dynamic Results component will be injected here via layout children */}
                        {React.Children.toArray(children)[0]}
                    </div>

                    {/* Right Column (3D Interactive Viewer) */}
                    <div className="w-2/3 glass rounded-3xl overflow-hidden relative shadow-2xl flex flex-col flex-grow min-h-[600px]">
                        {React.Children.toArray(children)[1]}
                    </div>

                </div>
            </main>
        </div>
    );
};

const NavItem = ({ icon, label, active }) => (
    <a href="#" className={`flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300
    ${active ? 'bg-primary text-white shadow-md shadow-primary/30' : 'text-slate-600 hover:bg-white hover:shadow-sm'}`}>
        {React.cloneElement(icon, { className: 'w-5 h-5' })}
        <span className="font-medium text-sm">{label}</span>
    </a>
);

export default DashboardLayout;
