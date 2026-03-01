/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: "#0EA5E9",
                secondary: "#1E293B",
                accent: "#38BDF8",
                danger: "#EF4444",
                success: "#10B981"
            }
        },
    },
    plugins: [],
}
