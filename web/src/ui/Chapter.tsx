import type { ReactNode } from "react";

interface Props {
  id: string;
  title: string;
  lead: ReactNode;
  children: ReactNode;
}

export function Chapter({ id, title, lead, children }: Props) {
  return (
    <section className="chapter" id={id} aria-labelledby={`${id}-h`}>
      <div className="wrap">
        <div className="chapter-head">
          <h2 id={`${id}-h`}>{title}</h2>
          <p className="lead">{lead}</p>
        </div>
        {children}
      </div>
    </section>
  );
}
