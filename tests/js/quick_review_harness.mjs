// Pure helpers behind the "quick review" naming mode (web/js/clusters.js).
import { installDom, webModule } from "./fake_dom.mjs";

installDom();
const { quickReviewCandidates, quickReviewAction } = await import(webModule("clusters.js"));

let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : (fail++, console.error("FAIL:", name)); };

// ---- candidates: the suggestion leads, then most-photographed people, max 6, no dupes
const people = [
  { id: 1, name: "Anna", photo_count: 10 }, { id: 2, name: "Bob", photo_count: 50 },
  { id: 3, name: "Cleo", photo_count: 5 }, { id: 4, name: "Dan", photo_count: 30 },
  { id: 5, name: "Eve", photo_count: 20 }, { id: 6, name: "Fay", photo_count: 1 },
  { id: 7, name: "Gus", photo_count: 99 },
];
let c = quickReviewCandidates(people, { person_id: 3, name: "Cleo", confidence: 0.8 });
check("suggestion first", c[0].id === 3);
check("max six", c.length === 6);
check("no duplicates", new Set(c.map((x) => x.id)).size === c.length);
check("then by popularity", c[1].id === 7 && c[2].id === 2);
c = quickReviewCandidates(people, null);
check("no suggestion -> most photographed first", c[0].id === 7);
check("empty people", quickReviewCandidates([], null).length === 0);

// ---- keys
const A = (k, n = 3) => quickReviewAction(k, n);
check("1 picks index 0", A("1").type === "pick" && A("1").index === 0);
check("3 picks index 2", A("3").index === 2);
check("key beyond candidates ignored", A("5", 3) === null);
check("n -> new person", A("n").type === "new" && A("N").type === "new");
check("s -> skip", A("s").type === "skip");
check("ArrowRight -> skip", A("ArrowRight").type === "skip");
check("z is not bound (no undo endpoint)", A("z") === null);
check("other keys ignored", A("q") === null && A("0") === null);

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
