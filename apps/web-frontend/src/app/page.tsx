"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";

import { Button } from "@/components/ui/button";
import { Nav } from "@/components/shared/nav";
import { siteConfig } from "@/lib/design";
import { useAuth } from "@/lib/auth-context";
import { DotBackground } from "@/components/shared/dot-background";

const STEPS = [
  {
    step: 1,
    title: "Sign up & sign in",
    description:
      "Create your account in seconds. Split-screen auth with a clean, minimal form.",
    image: "/screenshots/01-signin.png",
  },
  {
    step: 2,
    title: "Create or join a project",
    description:
      "Set up a new project with custom board columns and invite your team, or jump into an existing one.",
    image: "/screenshots/02-projects.png",
  },
  {
    step: 3,
    title: "Manage your board",
    description:
      "Create tickets, drag and drop between columns, and track progress with a full Kanban workflow.",
    image: "/screenshots/03-board.png",
  },
  {
    step: 4,
    title: "View your team",
    description:
      "See each member's workload, sprint capacity, and availability at a glance.",
    image: "/screenshots/04-team.png",
  },
  {
    step: 5,
    title: "AI-powered ticket recommendations",
    description:
      "Open any ticket to see suggested assignees ranked by semantic skill match, lexical overlap, and capacity.",
    image: "/screenshots/05-ticket-modal.png",
  },
  {
    step: 6,
    title: "Engineer recommendations",
    description:
      "Click any team member to see which unassigned tickets best match their skills and experience.",
    image: "/screenshots/06-member-modal.png",
  },
  {
    step: 7,
    title: "Upload your resume",
    description:
      "Upload a PDF or DOCX resume to build your engineer profile. AI extracts skills, generates embeddings, and enables cold-start recommendations.",
    image: "/screenshots/07-resume.png",
  },
];

export default function Home() {
  const { user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && user) {
      router.replace("/dashboard");
    }
  }, [user, isLoading, router]);

  if (isLoading || user) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Nav />
      {/* ========== HERO ========== */}
      <main className="flex flex-1 flex-col items-center px-6 text-center">
        <div className="mx-auto max-w-3xl space-y-8 pt-24 pb-16">
          <DotBackground />
          <div className="space-y-4">
            <div className="inline-block rounded-full border px-4 py-1.5 text-sm text-muted-foreground">
              AI-powered ticket assignment
            </div>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
              {siteConfig.tagline.split(".")[0]}.
              <br />
              <span className="text-muted-foreground">
                {siteConfig.tagline.split(".")[1]?.trim()}.
              </span>
            </h1>
            <p className="mx-auto max-w-xl text-lg text-muted-foreground">
              {siteConfig.subtitle}
            </p>
          </div>

          <div className="flex items-center justify-center gap-4">
            <Link href="/signup">
              <Button size="lg">Get started for free</Button>
            </Link>
            <Link href="/signin">
              <Button variant="outline" size="lg">
                Sign in
              </Button>
            </Link>
          </div>

          <div className="mx-auto grid max-w-2xl gap-6 pt-12 text-left sm:grid-cols-3">
            <div className="space-y-2 rounded-lg border p-4">
              <h3 className="font-semibold">Skill matching</h3>
              <p className="text-sm text-muted-foreground">
                Cosine similarity between ticket embeddings and engineer profiles
                finds the best match instantly.
              </p>
            </div>
            <div className="space-y-2 rounded-lg border p-4">
              <h3 className="font-semibold">Experience decay</h3>
              <p className="text-sm text-muted-foreground">
                Engineer profiles evolve with every closed ticket, keeping
                recommendations fresh and accurate.
              </p>
            </div>
            <div className="space-y-2 rounded-lg border p-4">
              <h3 className="font-semibold">Cold start ready</h3>
              <p className="text-sm text-muted-foreground">
                Upload a resume and get meaningful recommendations before an
                engineer has any ticket history.
              </p>
            </div>
          </div>
        </div>

        {/* ========== HOW IT WORKS ========== */}
        <section className="w-full border-t bg-muted/30 py-20">
          <div className="mx-auto max-w-6xl px-6">
            <div className="mb-16 text-center">
              <span className="inline-block rounded-full border px-4 py-1.5 text-sm text-muted-foreground mb-4">
                Step by step
              </span>
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                How it works
              </h2>
              <p className="mt-3 text-lg text-muted-foreground">
                From sign-up to AI-powered recommendations in six simple steps.
              </p>
            </div>

            <div className="space-y-24">
              {STEPS.map((item, idx) => {
                const isEven = idx % 2 === 0;
                return (
                  <div
                    key={item.step}
                    className={`flex flex-col items-center gap-8 lg:gap-12 ${
                      isEven ? "lg:flex-row" : "lg:flex-row-reverse"
                    }`}
                  >
                    {/* Text */}
                    <div className="flex-1 space-y-3 text-center lg:text-left">
                      <div className="inline-flex size-10 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                        {item.step}
                      </div>
                      <h3 className="text-xl font-semibold tracking-tight">
                        {item.title}
                      </h3>
                      <p className="max-w-md text-muted-foreground">
                        {item.description}
                      </p>
                    </div>

                    {/* Screenshot */}
                    <div className="flex-1">
                      <div className="overflow-hidden rounded-xl border bg-background shadow-lg transition-transform hover:scale-[1.02]">
                        <Image
                          src={item.image}
                          alt={item.title}
                          width={800}
                          height={500}
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* CTA */}
            <div className="mt-20 text-center">
              <h3 className="text-2xl font-bold tracking-tight">
                Ready to streamline your workflow?
              </h3>
              <p className="mt-2 text-muted-foreground">
                Get started in under a minute. No credit card required.
              </p>
              <div className="mt-6 flex items-center justify-center gap-4">
                <Link href="/signup">
                  <Button size="lg">Get started for free</Button>
                </Link>
                <Link href="/signin">
                  <Button variant="outline" size="lg">
                    Sign in
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-6 text-center text-sm text-muted-foreground">
        Built by the {siteConfig.name} team.
      </footer>
    </div>
  );
}
