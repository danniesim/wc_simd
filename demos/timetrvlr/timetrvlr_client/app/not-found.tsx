export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black p-12">
      <main className="max-w-xl w-full text-center">
        <h1 className="text-3xl font-semibold mb-4">Page not found</h1>
        <p className="mb-8 text-zinc-600 dark:text-zinc-300">
          The page you requested does not exist. Return to the home page.
        </p>
        <a
          href="/"
          className="inline-block rounded-md bg-black px-4 py-2 text-white shadow hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
        >
          Go Home
        </a>
      </main>
    </div>
  );
}
