import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <Image
          src="/timetrvlr.svg"
          alt="TIMETRVLR Logo"
          width={800}
          height={0}
          className="mb-16"
        />
        <div className="flex w-full items-center justify-center sm:justify-start">
          <Link
            href="/embed3d"
            className="rounded-md bg-black px-4 py-2 text-white shadow hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
          >
            Open 3D Embedding Viewer
          </Link>
        </div>
      </main>
    </div>
  );
}
