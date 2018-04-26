import React from "react";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";

const Grid = styled.div`
  display: grid;
  grid-gap: 1em;
  grid-template-columns: 1fr 2fr;
`;

const Avatar = styled(Img)`
  margin: auto;
  width: 150px;
  border-radius: 50%;
  border: 1px solid grey;
  @media (max-width: 600px) {
    width: 110px;
  }
`;

const H1 = styled.h1`
  align-self: center;
  color: black;
  margin: 0;
`;

export default ({ data }) => (
  <div>
    <Helmet>
      <title>Thanks</title>
    </Helmet>
    <Grid>
      <Avatar sizes={data.avatar.sizes} />
      <H1>Thanks for reaching out! We'll be in touch.</H1>
    </Grid>
  </div>
);

export const pageQuery = graphql`
  query ThanksQuery {
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
