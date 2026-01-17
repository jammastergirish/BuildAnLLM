"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const links = [
  { href: "/", label: "Overview" },
  { href: "/pretrain", label: "Pre-Training" },
  { href: "/finetune", label: "Fine-Tuning" },
  { href: "/inference", label: "Inference" },
  { href: "/evaluations", label: "Evaluations" },
];

const playgroundItems = [{ href: "/playground", label: "Tokenizer" }];

export default function Nav() {
  const pathname = usePathname();
  const isPlaygroundActive = pathname.startsWith("/playground");

  return (
    <nav className="primary">
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className={clsx({
            active: pathname === link.href,
          })}
        >
          {link.label}
        </Link>
      ))}
      <div className={clsx("nav-dropdown", { active: isPlaygroundActive })}>
        <button type="button" className={clsx("nav-dropdown-trigger", { active: isPlaygroundActive })}>
          Playground
        </button>
        <div className="nav-dropdown-menu">
          {playgroundItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={clsx("nav-dropdown-item", {
                active: pathname === item.href,
              })}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
