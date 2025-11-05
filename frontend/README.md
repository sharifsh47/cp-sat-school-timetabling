# Timetabling Solver Frontend (Next.js)

This is the frontend application for the timetabling solver, built with Next.js and Tailwind CSS. It allows users to input timetabling problem data in JSON format, send it to the FastAPI backend, and display the resulting schedule.

## Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    pnpm install
    # or npm install or yarn install, depending on your preference
    ```

## Running the Development Server

To start the Next.js development server, run:

```bash
pnpm dev
# or npm run dev or yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

The application will automatically reload if you make changes to the source files.

## Data Input Format

The frontend expects JSON data for the timetabling problem. An example of the expected format is pre-filled in the textarea, which you can modify. This JSON directly corresponds to the `ProblemData` Pydantic model in the FastAPI backend.

## Integration with Backend

This frontend communicates with the FastAPI backend running at `http://localhost:8000`. Ensure your backend server is running before attempting to solve a timetable.
