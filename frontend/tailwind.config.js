/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          50: '#FCFBF7',
          100: '#F7F5EC',
          200: '#EFECE0',
          300: '#E5E1D3',
        },
        ivory: '#FFFDF7',
        olive: {
          50: '#F4F6EE',
          100: '#E6EAD9',
          200: '#CCD5B5',
          300: '#A7AE8A', // muted sage
          400: '#7E8B54',
          500: '#606B3F', // primary olive
          600: '#4F5834',
          700: '#3F492C', // deep olive
          800: '#2E3520',
          900: '#1D2114',
        },
        sage: '#A7AE8A',
        charcoal: {
          DEFAULT: '#292D23', // primary text
          muted: '#77796D',   // secondary text
          subtle: '#9A9C92',
        },
        gain: {
          DEFAULT: '#2E6930',
          bg: '#EAF4EB',
          border: '#C3E2C6',
        },
        loss: {
          DEFAULT: '#B33927',
          bg: '#FAECE9',
          border: '#F0C7C0',
        },
        borderBase: '#E5E1D3',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(41, 45, 35, 0.04), 0 1px 2px 0 rgba(41, 45, 35, 0.02)',
        'card-hover': '0 4px 12px 0 rgba(41, 45, 35, 0.06), 0 2px 4px 0 rgba(41, 45, 35, 0.03)',
        'modal': '0 20px 25px -5px rgba(41, 45, 35, 0.1), 0 10px 10px -5px rgba(41, 45, 35, 0.04)',
      }
    },
  },
  plugins: [],
}
