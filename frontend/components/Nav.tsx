"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const links = [
  { href: "/", label: "Overview" },
  { href: "/pretrain", label: "Pre-Training" },
  { href: "/finetune", label: "Fine-Tuning" },
  { href: "/inference", label: "Inference" },
  { href: "/playground", label: "Playground" },
];

export default function Nav() {
  const pathname = usePathname();

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
    </nav>
  );
}
