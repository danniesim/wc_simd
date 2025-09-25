# Minimal Next.js Application

This is a minimal React application built using Next.js. It serves as a starting point for developing web applications with React and TypeScript.

## Project Structure

```
demos
└── imagetalk
    └── frontend
        ├── app
        │   ├── globals.css        # Global CSS styles
        │   ├── layout.tsx        # Layout component
        │   └── page.tsx          # Main entry point
        ├── components
        │   └── Header.tsx        # Header component
        ├── public
        │   └── .gitkeep          # Keeps the public directory in Git
        ├── package.json           # npm configuration
        ├── tsconfig.json          # TypeScript configuration
        ├── next.config.js         # Next.js configuration
        ├── .eslintrc.json         # ESLint configuration
        ├── .gitignore             # Git ignore file
        └── README.md              # Project documentation
```

## Getting Started

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd demos/imagetalk/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

4. **Open your browser**:
   Navigate to `http://localhost:3000` to see your application in action.

## Features

- **Global Styles**: Defined in `app/globals.css`.
- **Layout Component**: The layout for the application is managed in `app/layout.tsx`.
- **Header Component**: A reusable header component located in `components/Header.tsx`.

## Contributing

Feel free to submit issues or pull requests to improve this project. 

## License

This project is licensed under the MIT License.