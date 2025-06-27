# Adaptive Learning System Frontend

A beautiful, modern React frontend for the Quantum-Inspired Adaptive Learning System, built with shadcn/ui components and Tailwind CSS.

## Features

- **Modern UI Design**: Clean, accessible interface using shadcn/ui components
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Progress Tracking**: Visual progress bars and charts for knowledge mastery
- **Interactive Question Interface**: Beautiful radio button alternatives with smooth animations
- **Gradient Backgrounds**: Subtle gradients and backdrop blur effects
- **Loading States**: Elegant loading animations with icons
- **Feedback System**: Clear success/error feedback with appropriate styling

## Tech Stack

- **React 19**: Latest React with hooks and modern patterns
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Beautiful, accessible component library
- **Lucide React**: Modern icon library
- **Recharts**: Responsive chart library for progress visualization
- **Axios**: HTTP client for API communication

## Components Used

### shadcn/ui Components
- **Button**: Multiple variants (default, outline, ghost, etc.)
- **Card**: Structured content containers with headers and content areas
- **Progress**: Animated progress bars for knowledge tracking
- **Badge**: Small status indicators
- **Alert**: Feedback messages with different severity levels

### Custom Styling
- **Gradient Backgrounds**: Subtle blue-to-purple gradients
- **Backdrop Blur**: Modern glass-morphism effects
- **Smooth Animations**: Hover effects and transitions
- **Responsive Grid**: Adaptive layout for different screen sizes

## Getting Started

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm start
   ```

3. **Build for Production**:
   ```bash
   npm run build
   ```

## API Integration

The frontend communicates with the Flask backend API:

- `GET /api/question` - Fetch current question
- `POST /api/submit` - Submit answer and get feedback
- `POST /api/reset` - Reset user session
- `GET /api/stats` - Get learning statistics

## Design System

### Colors
- **Primary**: Blue (#3b82f6) to Purple (#8b5cf6) gradients
- **Success**: Green (#10b981) for correct answers
- **Error**: Red (#ef4444) for incorrect answers
- **Background**: Subtle gradients with white/transparent overlays

### Typography
- **Headings**: Bold, large text for hierarchy
- **Body**: Readable, medium-weight text
- **Captions**: Small, muted text for secondary information

### Spacing
- Consistent 8px grid system
- Generous padding and margins for breathing room
- Proper spacing between interactive elements

## Accessibility

- **Screen Reader Support**: Proper ARIA labels and semantic HTML
- **Keyboard Navigation**: Full keyboard accessibility
- **Color Contrast**: WCAG AA compliant color combinations
- **Focus Indicators**: Clear focus states for all interactive elements

## Performance

- **Optimized Build**: Tree-shaking and code splitting
- **Lazy Loading**: Components load only when needed
- **Efficient Re-renders**: Proper React optimization patterns
- **Minimal Bundle Size**: Only essential dependencies included

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Development

### File Structure
```
src/
├── components/
│   └── ui/           # shadcn/ui components
├── lib/
│   └── utils.js      # Utility functions
├── App.js            # Main application component
└── index.css         # Global styles and Tailwind imports
```

### Adding New Components

1. Create component in `src/components/ui/`
2. Follow shadcn/ui patterns with proper TypeScript-like props
3. Use the `cn()` utility for className merging
4. Export with proper displayName for debugging

### Styling Guidelines

- Use Tailwind CSS classes for styling
- Leverage CSS custom properties for theming
- Follow the established color palette
- Maintain consistent spacing and typography

## Contributing

1. Follow the existing code style and patterns
2. Ensure all components are accessible
3. Test on multiple screen sizes
4. Update documentation for new features

## License

This project is part of the Adaptive Learning System and follows the same license terms.
