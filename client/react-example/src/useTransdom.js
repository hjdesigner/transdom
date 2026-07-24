import { useEffect, useRef, useState } from "react";
import { Transdom } from "../../transdom.js";

export function useTransdom(config, rootRef) {
  const instanceRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    const transdom = new Transdom({
      ...config,
      root: rootRef.current || document.body,
    });
    instanceRef.current = transdom;

    const handleStart = () => {
      setStatus("loading");
      setError(null);
    };
    const handleSuccess = () => setStatus("success");
    const handleError = (event) => {
      setStatus("error");
      setError(event.detail.error);
    };

    transdom.addEventListener("translate:start", handleStart);
    transdom.addEventListener("translate:success", handleSuccess);
    transdom.addEventListener("translate:error", handleError);

    return () => {
      transdom.stopAutoTranslate();
      transdom.removeEventListener("translate:start", handleStart);
      transdom.removeEventListener("translate:success", handleSuccess);
      transdom.removeEventListener("translate:error", handleError);
    };
  }, []);

  const translate = () => instanceRef.current?.startAutoTranslate();
  const stop = () => instanceRef.current?.stopAutoTranslate();

  return { status, error, translate, stop };
}