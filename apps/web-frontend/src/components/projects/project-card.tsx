"use client";

import Link from "next/link";
import { Users, Folder, ArrowRight } from "lucide-react";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ProjectListItem } from "@/lib/api";

interface ProjectCardProps {
  project: ProjectListItem;
}

export function ProjectCard({ project }: ProjectCardProps) {
  return (
    <Link href={`/projects/${project.slug}`} className="block h-full outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-xl">
      <Card className="group relative flex h-full flex-col overflow-hidden border-border/50 bg-card transition-all duration-300 hover:-translate-y-[2px] hover:border-primary/30 hover:shadow-md dark:hover:shadow-none dark:hover:bg-accent/5">
        
        <CardHeader className="p-5 pb-3">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              {/* Project Icon Anchor */}
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border/50 bg-muted/30 text-muted-foreground transition-colors group-hover:bg-primary/10 group-hover:text-primary group-hover:border-primary/20">
                <Folder className="size-4.5" />
              </div>
              
              {/* Title */}
              <CardTitle className="text-base font-bold tracking-tight text-foreground transition-colors group-hover:text-primary line-clamp-1">
                {project.name}
              </CardTitle>
            </div>

            {/* Role Badge */}
            <Badge 
              variant="secondary" 
              className={cn(
                "shrink-0 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                project.role === "owner" 
                  ? "bg-primary/10 text-primary hover:bg-primary/20" 
                  : "bg-muted text-muted-foreground"
              )}
            >
              {project.role}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="flex flex-1 flex-col p-5 pt-0">
          {/* Description area with fixed minimum height for grid consistency */}
          <p className="text-sm font-medium text-muted-foreground/80 line-clamp-2 min-h-[40px] mb-5">
            {project.description || (
              <span className="italic opacity-50">No description provided.</span>
            )}
          </p>
          
          {/* Footer block */}
          <div className="mt-auto flex items-center justify-between border-t border-border/40 pt-4">
            {/* Member Count Pill */}
            <div className="flex items-center gap-1.5 rounded-md bg-muted/40 px-2.5 py-1 text-xs font-semibold text-muted-foreground transition-colors group-hover:bg-muted/80">
              <Users className="size-3.5 opacity-70" />
              <span>
                {project.member_count} {project.member_count === 1 ? "Member" : "Members"}
              </span>
            </div>
            
            {/* Hover Action Indicator */}
            <div className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-primary opacity-0 transition-all duration-300 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0">
              Open Project
              <ArrowRight className="size-3" />
            </div>
          </div>
        </CardContent>
        
      </Card>
    </Link>
  );
}