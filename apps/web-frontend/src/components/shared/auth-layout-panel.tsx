import Image from "next/image";
import { Logo } from "@/components/shared/logo";
import { branding } from "@/lib/design";

export function AuthLayoutPanel() {
  return (
    <div className="relative hidden bg-primary lg:flex lg:flex-col lg:items-center lg:justify-between lg:p-10">
      <Logo variant="light" />

      <div className="flex flex-1 items-center justify-center">
        {/* Light mode: black anvil */}
        <Image
          src="/forge_dark.jpeg"
          alt="TicketForge"
          width={400}
          height={400}
          className="block dark:hidden drop-shadow-lg"
          priority
        />
        {/* Dark mode: white anvil */}
        <Image
          src="/forge_light.jpeg"
          alt="TicketForge"
          width={400}
          height={400}
          className="hidden dark:block drop-shadow-lg"
          priority
        />
      </div>

      <blockquote className="space-y-2">
        <p className="text-lg text-primary-foreground/80">
          &ldquo;{branding.testimonial.quote}&rdquo;
        </p>
        <footer className="text-sm text-primary-foreground/60">
          — {branding.testimonial.author}
        </footer>
      </blockquote>
    </div>
  );
}
